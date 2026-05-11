"""
engine.py
=========
Dynamic Pipeline Execution Engine for the Astrikos framework.

Accepts a JSON pipeline definition (a list of steps), validates each step
against the metadata registry, dynamically resolves functions from
nexus_blocks.py via ``getattr``, and executes them sequentially — passing
the output of one step as the input to the next.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nexus_blocks

# ── Paths ────────────────────────────────────────────────────────────
METADATA_PATH = Path(__file__).parent / "metadata.json"


# ── Helper: load the metadata registry ───────────────────────────────

def _load_metadata() -> dict:
    """Read and return the metadata registry from disk."""
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Core Engine ──────────────────────────────────────────────────────

class PipelineEngine:
    """Execute an ordered list of pipeline steps defined as JSON.

    Parameters
    ----------
    pipeline_json : list[dict]
        Each dict represents one step and must contain:
            - ``"block"``  : str   — function name (must exist in nexus_blocks)
            - ``"params"`` : dict  — keyword arguments to pass to the function

    Example pipeline_json::

        [
            {"block": "load_csv",              "params": {"filepath": "data/dummy_electricity_data.csv"}},
            {"block": "impute_missing_values",  "params": {"strategy": "mean"}},
            {"block": "train_random_forest",    "params": {"target_col": "kilowatt_hours"}}
        ]
    """

    def __init__(self, pipeline_json: list[dict]) -> None:
        self.pipeline = pipeline_json
        self.metadata = _load_metadata()
        self.registry = self.metadata.get("blocks", {})
        self.results: list[Any] = []

    # ── Validation ───────────────────────────────────────────────────

    def _validate_block_exists(self, block_name: str, step_num: int) -> None:
        """Ensure the block name exists in both the registry and the module."""
        # Check metadata registry
        if block_name not in self.registry:
            raise ValueError(
                f"[Step {step_num}] Block '{block_name}' not found in metadata.json registry. "
                f"Available blocks: {list(self.registry.keys())}"
            )
        # Check actual Python module
        if not hasattr(nexus_blocks, block_name):
            raise AttributeError(
                f"[Step {step_num}] Function '{block_name}' exists in metadata.json but is "
                f"missing from nexus_blocks.py module."
            )

    def _validate_parameters(self, block_name: str, params: dict, step_num: int) -> None:
        """Validate supplied params against the metadata schema."""
        block_meta = self.registry[block_name]
        meta_params = block_meta.get("parameters", {})
        meta_inputs = block_meta.get("inputs", {})

        # Collect all accepted parameter names (inputs + configurable parameters)
        # Exclude 'df' since it is passed implicitly between steps
        accepted = set(meta_inputs.keys()) | set(meta_params.keys()) - {"df"}

        # Check for unknown parameters
        for key in params:
            if key not in accepted:
                print(
                    f"  [WARN] Step {step_num} ('{block_name}'): "
                    f"Unknown parameter '{key}' — not in metadata schema."
                )

        # Validate allowed values for constrained parameters
        for key, value in params.items():
            if key in meta_params:
                allowed = meta_params[key].get("allowed_values")
                if allowed is not None and value not in allowed:
                    raise ValueError(
                        f"[Step {step_num}] Parameter '{key}' value '{value}' is invalid for "
                        f"block '{block_name}'. Allowed: {allowed}"
                    )

    def _validate_sequence(self, block_name: str, prev_block: str | None, step_num: int) -> None:
        """Ensure the block ordering respects valid_previous_blocks constraints."""
        block_meta = self.registry[block_name]
        valid_prev = block_meta.get("constraints", {}).get("valid_previous_blocks", [])

        # If the constraint list is empty, any predecessor (or none) is allowed
        if valid_prev and prev_block not in valid_prev:
            print(
                f"  [WARN] Step {step_num} ('{block_name}'): Previous block '{prev_block}' "
                f"is not in valid_previous_blocks {valid_prev}."
            )

    # ── Execution ────────────────────────────────────────────────────

    def execute(self) -> Any:
        """Run the full pipeline and return the final output.

        Returns
        -------
        Any
            The output of the last step (could be a DataFrame, model, etc.).
        """
        print("=" * 65)
        print("  PIPELINE ENGINE — Starting execution")
        print(f"  Total steps: {len(self.pipeline)}")
        print("=" * 65)

        carry: Any = None          # output carried between steps
        prev_block: str | None = None

        for idx, step in enumerate(self.pipeline, start=1):
            block_name: str = step.get("block", "")
            params: dict = step.get("params", {})

            print(f"\n{'-' * 65}")
            print(f"  Step {idx}/{len(self.pipeline)}: {block_name}")
            print(f"  Params: {params}")
            print(f"{'-' * 65}")

            # 1. Validate
            self._validate_block_exists(block_name, idx)
            self._validate_parameters(block_name, params, idx)
            self._validate_sequence(block_name, prev_block, idx)

            # 2. Resolve function dynamically
            func = getattr(nexus_blocks, block_name)

            # 3. Execute — inject the carried DataFrame if the block expects one
            try:
                if carry is None:
                    # First block (e.g., load_csv) — no DataFrame to inject
                    result = func(**params)
                else:
                    # Subsequent blocks — pass carried output as first argument
                    result = func(carry, **params)
            except Exception as exc:
                print(f"\n  [ERROR] Step {idx} ('{block_name}') failed: {exc}")
                raise RuntimeError(
                    f"Pipeline halted at step {idx} ('{block_name}'): {exc}"
                ) from exc

            # 4. Store result and carry forward
            self.results.append(result)
            carry = result
            prev_block = block_name

            print(f"  [OK] Step {idx} completed successfully.")

        print(f"\n{'=' * 65}")
        print("  PIPELINE ENGINE — All steps completed")
        print(f"{'=' * 65}")
        
        # 5. Save Artifacts
        artifacts = self._save_artifacts()
        
        return carry, artifacts

    def _save_artifacts(self) -> dict:
        import joblib
        import pandas as pd
        from datetime import datetime
        
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = {}

        # Find the last model and the last dataframe in the results
        final_model = None
        final_df = None
        
        for res in reversed(self.results):
            if isinstance(res, pd.DataFrame) and final_df is None:
                final_df = res
            elif hasattr(res, "predict") and final_model is None:
                final_model = res
                
        if final_model is not None:
            model_filename = f"model_{timestamp}.pkl"
            model_path = out_dir / model_filename
            joblib.dump(final_model, model_path)
            saved_paths["model_url"] = f"/outputs/{model_filename}"
            print(f"  [ARTIFACT] Saved model to {model_path}")
            
        if final_df is not None:
            df_filename = f"processed_data_{timestamp}.csv"
            df_path = out_dir / df_filename
            final_df.to_csv(df_path, index=False)
            saved_paths["dataframe_url"] = f"/outputs/{df_filename}"
            print(f"  [ARTIFACT] Saved processed data to {df_path}")
            
        return saved_paths


# ── Convenience runner ───────────────────────────────────────────────

def run_pipeline(pipeline_json: list[dict]) -> Any:
    """Shortcut: create an engine and execute in one call."""
    engine = PipelineEngine(pipeline_json)
    return engine.execute()


# ── CLI quick test ───────────────────────────────────────────────────

if __name__ == "__main__":
    sample_pipeline = [
        {
            "block": "load_csv",
            "params": {"filepath": "data/dummy_electricity_data.csv"},
        },
        {
            "block": "impute_missing_values",
            "params": {"strategy": "mean"},
        },
        {
            "block": "train_random_forest",
            "params": {"target_col": "kilowatt_hours"},
        },
    ]

    final_output, artifacts = run_pipeline(sample_pipeline)
    print(f"\nFinal output type: {type(final_output).__name__}")
    print(f"Artifacts: {artifacts}")
