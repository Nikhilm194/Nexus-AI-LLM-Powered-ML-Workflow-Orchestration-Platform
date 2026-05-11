"""
llm_orchestrator.py
===================
AI / LLM Layer for the Astrikos framework.

Sends a structured prompt to a local Ollama server (llama3) containing
the user intent, data profile, and metadata registry.  The LLM returns
a valid JSON pipeline which is parsed and handed to the execution engine.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from textwrap import dedent

import requests

# ── Configuration ────────────────────────────────────────────────────
OLLAMA_BASE    = "http://localhost:11434"
OLLAMA_URL     = f"{OLLAMA_BASE}/api/generate"
OLLAMA_MODEL   = "llama3"
CONNECT_TIMEOUT = 10     # seconds to establish TCP connection
READ_TIMEOUT    = 600    # seconds to wait for the LLM to finish generating
MAX_RETRIES     = 2      # retry once on timeout before falling back
METADATA_PATH = Path(__file__).parent / "metadata.json"


def _load_metadata() -> dict:
    """Read the block metadata registry from disk."""
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Hardcoded fallback (used when Ollama is unreachable) ─────────────

_FALLBACK_PIPELINE = [
    {"block": "load_csv",             "params": {"filepath": "data/dummy_electricity_data.csv"}},
    {"block": "impute_missing_values", "params": {"strategy": "mean"}},
    {"block": "train_random_forest",   "params": {"target_col": "kilowatt_hours"}},
]


# ── Main entry point ────────────────────────────────────────────────

def generate_pipeline_json(
    data_profile: dict,
    user_intent: str,
    metadata: dict | None = None,
    csv_path: str | None = None,
) -> list[dict]:
    """Call the local Ollama LLM and return a pipeline configuration.

    Parameters
    ----------
    data_profile : dict
        Output of ``data_profiler.generate_profile()``.
    user_intent : str
        Natural-language description of what the user wants to achieve.
    metadata : dict | None
        The block registry.  Loaded from ``metadata.json`` if *None*.
    csv_path : str | None
        Path to the CSV file used for data ingestion.  If provided, it
        will be injected into the ``load_csv`` step when the LLM omits it.

    Returns
    -------
    list[dict]
        An ordered list of pipeline steps, each with ``block`` and ``params``.
    """
    if metadata is None:
        metadata = _load_metadata()

    # ── 1. Build the prompt ──────────────────────────────────────────
    prompt = _build_system_prompt(data_profile, user_intent, metadata)

    print("\n" + "=" * 70)
    print("  LLM ORCHESTRATOR -- Prompt built (%d chars)" % len(prompt))
    print("=" * 70)
    print(prompt)
    print("=" * 70)

    # ── 1b. Warmup: ensure the model is loaded into memory ──────────
    _warmup_model()

    # ── 2. Call Ollama ───────────────────────────────────────────────
    payload = {
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "format":  "json",
        "options": {
            "num_predict": 512,      # cap tokens – pipeline JSON is small
            "temperature": 0.0,      # deterministic output
        },
    }

    print(f"\n[LLM ORCHESTRATOR] Sending request to Ollama ({OLLAMA_URL})...")
    print(f"                   Model: {OLLAMA_MODEL}  |  format: json  |  stream: False")
    print(f"                   Timeout: connect={CONNECT_TIMEOUT}s  read={READ_TIMEOUT}s")

    resp = _call_ollama_with_retry(payload)

    if resp is None:
        return _FALLBACK_PIPELINE

    # ── 3. Extract the raw LLM text ─────────────────────────────────
    raw_response = resp.json().get("response", "")
    print(f"\n[LLM ORCHESTRATOR] Raw LLM response ({len(raw_response)} chars):")
    print(raw_response)

    # ── 4. Parse JSON ────────────────────────────────────────────────
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        print(f"\n[LLM ORCHESTRATOR] ERROR: Failed to parse LLM output as JSON: {exc}")
        print("[LLM ORCHESTRATOR] Falling back to hardcoded pipeline.")
        return _FALLBACK_PIPELINE

    # The LLM might return {"pipeline": [...]} or just [...]
    if isinstance(parsed, dict):
        # Try common wrapper keys first
        for key in ("pipeline", "steps", "blocks",
                    "execution_pipeline_steps", "pipeline_steps"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            # Generic fallback: grab the first value that is a list
            for key, val in parsed.items():
                if isinstance(val, list):
                    print(f"[LLM ORCHESTRATOR] INFO: Unwrapping from key '{key}'")
                    parsed = val
                    break
            else:
                print("[LLM ORCHESTRATOR] WARN: LLM returned a dict with no list value.")
                print("[LLM ORCHESTRATOR] Falling back to hardcoded pipeline.")
                return _FALLBACK_PIPELINE

    if not isinstance(parsed, list):
        print("[LLM ORCHESTRATOR] WARN: Parsed output is not a list.")
        print("[LLM ORCHESTRATOR] Falling back to hardcoded pipeline.")
        return _FALLBACK_PIPELINE

    # ── 5. Post-process: inject any missing required params ─────────
    parsed = _postprocess_pipeline(parsed, metadata, csv_path)

    print(f"\n[LLM ORCHESTRATOR] Successfully parsed {len(parsed)} pipeline steps:")
    for i, step in enumerate(parsed, 1):
        print(f"  Step {i}: {step.get('block', '?')}  |  params={step.get('params', {})}")

    return parsed


# ── Post-processing ─────────────────────────────────────────────────

def _postprocess_pipeline(
    pipeline: list[dict],
    metadata: dict,
    csv_path: str | None,
) -> list[dict]:
    """Fill in missing required params that the LLM is known to omit.

    The LLM treats ``filepath`` as a data-flow "input" (which it is in the
    metadata registry) rather than a runtime "param", so it returns
    ``load_csv`` with empty params.  This function patches that gap.
    """
    registry = metadata.get("blocks", {})

    for step in pipeline:
        block_name = step.get("block", "")
        params = step.setdefault("params", {})
        block_meta = registry.get(block_name, {})

        # --- Inject filepath for data-ingestion blocks ---
        if block_name == "load_csv" and "filepath" not in params:
            if csv_path:
                params["filepath"] = csv_path
                print(f"[LLM ORCHESTRATOR] POST-PROCESS: Injected filepath="
                      f"'{csv_path}' into '{block_name}' params.")
            else:
                # Try to infer from the example_usage in metadata
                example = block_meta.get("example_usage", "")
                if "'" in example:
                    inferred = example.split("'")[1]
                    params["filepath"] = inferred
                    print(f"[LLM ORCHESTRATOR] POST-PROCESS: Inferred filepath="
                          f"'{inferred}' from metadata example.")

        # --- Inject required params that have defaults in metadata ---
        for param_name, param_meta in block_meta.get("parameters", {}).items():
            if param_name not in params:
                default = param_meta.get("default")
                if default is not None:
                    params[param_name] = default
                    print(f"[LLM ORCHESTRATOR] POST-PROCESS: Injected {param_name}="
                          f"'{default}' (default) into '{block_name}'.")

    return pipeline


# ── Helpers ──────────────────────────────────────────────────────────

def _warmup_model() -> None:
    """Send a tiny throwaway request so Ollama loads the model into RAM/VRAM.

    The first real request after a cold start is much slower because the
    model weights must be read from disk.  This no-op generation forces
    the load to happen before the big prompt.
    """
    print("[LLM ORCHESTRATOR] Warming up model (loading into memory)...")
    try:
        warmup_payload = {
            "model":  OLLAMA_MODEL,
            "prompt": "hi",
            "stream": False,
            "options": {"num_predict": 1},   # generate 1 token only
        }
        r = requests.post(OLLAMA_URL, json=warmup_payload,
                          timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        r.raise_for_status()
        print("[LLM ORCHESTRATOR] Model loaded and warm.")
    except Exception as exc:
        print(f"[LLM ORCHESTRATOR] WARN: Warmup failed ({exc}), continuing anyway.")


def _call_ollama_with_retry(payload: dict) -> requests.Response | None:
    """POST to Ollama with automatic retry on timeout."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[LLM ORCHESTRATOR] Attempt {attempt}/{MAX_RETRIES}...")
            t0 = time.time()
            resp = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            elapsed = time.time() - t0
            resp.raise_for_status()
            print(f"[LLM ORCHESTRATOR] Ollama responded in {elapsed:.1f}s")
            return resp
        except requests.ConnectionError:
            print(f"[LLM ORCHESTRATOR] ERROR: Cannot reach Ollama at {OLLAMA_URL}")
            break  # no point retrying a connection failure
        except requests.Timeout:
            print(f"[LLM ORCHESTRATOR] ERROR: Attempt {attempt} timed out "
                  f"(connect={CONNECT_TIMEOUT}s, read={READ_TIMEOUT}s).")
            if attempt < MAX_RETRIES:
                print("[LLM ORCHESTRATOR] Retrying...")
                continue
        except requests.HTTPError as exc:
            print(f"[LLM ORCHESTRATOR] ERROR: HTTP {resp.status_code}: {exc}")
            break

    print("[LLM ORCHESTRATOR] Falling back to hardcoded pipeline.")
    return None


# ── Prompt construction ──────────────────────────────────────────────

def _build_system_prompt(
    data_profile: dict,
    user_intent: str,
    metadata: dict,
) -> str:
    """Assemble the full prompt string for the Ollama LLM."""

    # Format the data profile section
    profile_lines = [
        f"  Rows   : {data_profile['row_count']}",
        f"  Columns: {data_profile['column_count']}",
        "",
    ]
    for col in data_profile["columns"]:
        nan_flag = f"  ** {col['null_count']} missing ({col['null_pct']}%) **" if col["null_count"] > 0 else ""
        profile_lines.append(
            f"  - {col['name']:20s}  type={col['dtype']:15s}  nulls={col['null_count']}{nan_flag}"
        )
    profile_section = "\n".join(profile_lines)

    # Format the metadata / block registry section
    blocks_section_parts = []
    for block_name, block_meta in metadata.get("blocks", {}).items():
        inputs_str = json.dumps(block_meta.get("inputs", {}), indent=4)
        outputs_str = json.dumps(block_meta.get("outputs", {}), indent=4)
        params_str = json.dumps(block_meta.get("parameters", {}), indent=4)
        constraints_str = json.dumps(block_meta.get("constraints", {}), indent=4)

        blocks_section_parts.append(dedent(f"""\
            ### Block: {block_name}
            Display Name : {block_meta.get('display_name', '')}
            Category     : {block_meta.get('category', '')}
            Description  : {block_meta.get('description', '')}

            Inputs:
            {inputs_str}

            Outputs:
            {outputs_str}

            Configurable Parameters:
            {params_str}

            Constraints & Rules:
            {constraints_str}

            Example: {block_meta.get('example_usage', '')}
        """))

    blocks_section = "\n".join(blocks_section_parts)

    # Assemble the full prompt
    prompt = dedent(f"""\
        You are an ML architect. Based on this data profile and metadata registry,
        output ONLY a JSON array representing the execution pipeline steps.
        Do not include markdown formatting or explanations.

        =====================================================================
        SECTION 1: USER INTENT
        =====================================================================
        The user said:
        "{user_intent}"

        =====================================================================
        SECTION 2: DATA PROFILE (auto-generated from the uploaded dataset)
        =====================================================================
        {profile_section}

        =====================================================================
        SECTION 3: AVAILABLE PIPELINE BLOCKS (from metadata registry)
        =====================================================================
        Below is every block you may use.  You MUST respect each block's
        inputs, outputs, parameters, and ordering constraints.

        {blocks_section}

        =====================================================================
        SECTION 4: RESPONSE FORMAT
        =====================================================================
        Return ONLY a valid JSON array. Each element must be an object with:
          - "block"  : the exact function_name from the registry above
          - "params" : a dict of keyword arguments to pass to that function

        Rules:
        1. The first block must be a data_ingestion block (e.g., load_csv).
        2. If the data profile shows missing values, include an imputation block
           BEFORE any model_building block.
        3. Respect valid_previous_blocks / valid_next_blocks constraints.
        4. Only use parameter values that are listed in allowed_values (if set).
        5. Do NOT include any explanation -- return raw JSON only.

        Example response:
        [
          {{"block": "load_csv",             "params": {{"filepath": "data/file.csv"}}}},
          {{"block": "impute_missing_values", "params": {{"strategy": "mean"}}}},
          {{"block": "train_random_forest",   "params": {{"target_col": "target"}}}}
        ]
    """)

    return prompt


# ── CLI quick test ───────────────────────────────────────────────────

if __name__ == "__main__":
    from data_profiler import generate_profile
    import pandas as pd

    # 1. Load & profile
    csv_path = "data/dummy_electricity_data.csv"
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    profile = generate_profile(df)

    # 2. Simulate user intent
    user_intent = "Can you give me an ML model for electricity consumption?"

    # 3. Generate pipeline via Ollama
    pipeline = generate_pipeline_json(profile, user_intent, csv_path=csv_path)

    print("\n[FINAL PIPELINE JSON]")
    print(json.dumps(pipeline, indent=2))
