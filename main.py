"""
main.py
=======
FastAPI backend for the Astrikos AI-driven ML pipeline framework.

Endpoints
---------
POST /api/profile   – Upload a CSV → get a data profile JSON.
POST /api/generate  – Send profile + intent → get LLM-generated pipeline JSON.
POST /api/execute   – Send pipeline JSON + CSV → run the pipeline via the engine.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_profiler import generate_profile
from llm_orchestrator import generate_pipeline_json
from engine import PipelineEngine
from chat_orchestrator import generate_chat_response

# ── App setup ────────────────────────────────────────────────────────

app = FastAPI(
    title="Astrikos API",
    description="AI-driven drag-and-drop ML pipeline framework",
    version="1.0.0",
)

# CORS — allow all origins so a future frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static output artifacts
Path("outputs").mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


# ── Request / Response schemas ───────────────────────────────────────

class ColumnProfile(BaseModel):
    """Schema for a single column in the data profile."""
    name: str
    dtype: str
    null_count: int
    null_pct: float

    model_config = {"json_schema_extra": {
        "example": {
            "name": "temperature",
            "dtype": "float64",
            "null_count": 5,
            "null_pct": 5.0,
        }
    }}


class DataProfile(BaseModel):
    """Schema for the full data profile returned by /api/profile."""
    row_count: int = Field(..., description="Total number of rows", examples=[100])
    column_count: int = Field(..., description="Total number of columns", examples=[4])
    columns: list[ColumnProfile] = Field(
        ..., description="Per-column profiling info",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "row_count": 100,
            "column_count": 4,
            "columns": [
                {"name": "timestamp",      "dtype": "datetime64[ns]", "null_count": 0, "null_pct": 0.0},
                {"name": "temperature",    "dtype": "float64",        "null_count": 5, "null_pct": 5.0},
                {"name": "humidity",       "dtype": "float64",        "null_count": 0, "null_pct": 0.0},
                {"name": "kilowatt_hours", "dtype": "float64",        "null_count": 0, "null_pct": 0.0},
            ],
        }
    }}


class GenerateRequest(BaseModel):
    """Body for the /api/generate endpoint."""
    data_profile: DataProfile = Field(
        ..., description="Output from /api/profile",
    )
    user_intent: str = Field(
        ..., description="Natural-language intent",
        examples=["Clean the dataset and predict kilowatt_hours."],
    )

    model_config = {"json_schema_extra": {
        "example": {
            "data_profile": {
                "row_count": 100,
                "column_count": 4,
                "columns": [
                    {"name": "timestamp",      "dtype": "datetime64[ns]", "null_count": 0, "null_pct": 0.0},
                    {"name": "temperature",    "dtype": "float64",        "null_count": 5, "null_pct": 5.0},
                    {"name": "humidity",       "dtype": "float64",        "null_count": 0, "null_pct": 0.0},
                    {"name": "kilowatt_hours", "dtype": "float64",        "null_count": 0, "null_pct": 0.0},
                ],
            },
            "user_intent": "Clean the dataset and predict kilowatt_hours.",
        }
    }}


class PipelineStep(BaseModel):
    """A single step in the pipeline."""
    block: str = Field(..., description="Block function name", examples=["load_csv"])
    params: dict = Field(default_factory=dict, description="Keyword arguments")


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Role of the sender (e.g., 'user', 'assistant')")
    content: str = Field(..., description="The message content")

class ChatRequest(BaseModel):
    """Body for the /api/chat endpoint."""
    messages: list[ChatMessage] = Field(..., description="List of previous messages in the conversation")


# ── Endpoints ────────────────────────────────────────────────────────

@app.post("/api/profile", summary="Profile a CSV file")
async def profile_csv(file: UploadFile = File(...)):
    """Accept a CSV upload, profile it with ``data_profiler``, return JSON.

    Parameters
    ----------
    file : UploadFile
        The CSV file to profile.

    Returns
    -------
    dict
        ``{ "profile": { row_count, column_count, columns: [...] } }``
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), parse_dates=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}")

    profile = generate_profile(df)
    return {"profile": profile, "filename": file.filename}


@app.post("/api/generate", summary="Generate pipeline via LLM")
async def generate_pipeline(body: GenerateRequest):
    """Send a data profile and user intent to the Ollama LLM orchestrator.

    The LLM returns a pipeline definition as a JSON array of steps.

    Parameters
    ----------
    body : GenerateRequest
        JSON body with ``data_profile`` (DataProfile) and ``user_intent`` (str).

    Returns
    -------
    dict
        ``{ "pipeline": [ { "block": ..., "params": ... }, ... ] }``
    """
    # Convert the validated Pydantic model back to a plain dict
    # so the orchestrator gets the same format it expects.
    profile_dict = body.data_profile.model_dump()

    try:
        pipeline = generate_pipeline_json(
            data_profile=profile_dict,
            user_intent=body.user_intent,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM orchestrator error: {exc}",
        )

    return {"pipeline": pipeline}


_EXECUTE_PIPELINE_EXAMPLE = json.dumps([
    {"block": "load_csv",             "params": {"filepath": "data/file.csv"}},
    {"block": "impute_missing_values", "params": {"strategy": "mean"}},
    {"block": "train_random_forest",   "params": {"target_col": "kilowatt_hours"}},
])


@app.post("/api/execute", summary="Execute a pipeline")
async def execute_pipeline(
    file: UploadFile = File(..., description="The CSV dataset to run the pipeline on."),
    pipeline_json: str = Form(
        ...,
        description="A JSON array of pipeline step objects.",
        examples=[_EXECUTE_PIPELINE_EXAMPLE],
    ),
):
    """Execute a pipeline against an uploaded CSV file.

    Parameters
    ----------
    file : UploadFile
        The CSV dataset to run the pipeline on.
    pipeline_json : str
        A JSON string representing the pipeline (list of step objects).

    Returns
    -------
    dict
        Execution summary including step results and final output type.
    """
    # ── 1. Validate pipeline_json is not empty ───────────────────────
    if not pipeline_json or not pipeline_json.strip():
        raise HTTPException(
            status_code=400,
            detail="pipeline_json is required. Provide a JSON array of pipeline steps, e.g.: "
                   + _EXECUTE_PIPELINE_EXAMPLE,
        )

    # ── 2. Parse the pipeline JSON string ────────────────────────────
    try:
        pipeline = json.loads(pipeline_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid pipeline JSON: {exc}")

    if not isinstance(pipeline, list):
        raise HTTPException(status_code=400, detail="Pipeline must be a JSON array.")

    if len(pipeline) == 0:
        raise HTTPException(status_code=400, detail="Pipeline array is empty.")

    # ── 3. Validate each step has a 'block' field ────────────────────
    for i, step in enumerate(pipeline):
        if "block" not in step:
            raise HTTPException(
                status_code=400,
                detail=f"Pipeline step {i} is missing the required 'block' field.",
            )

    # ── 4. Save uploaded CSV to a temporary file ─────────────────────
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    try:
        contents = await file.read()
        tmp_dir = Path(tempfile.gettempdir())
        tmp_csv = tmp_dir / f"astrikos_{file.filename}"
        tmp_csv.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}")

    # ── 5. Patch the load_csv step to use the temp file path ─────────
    for step in pipeline:
        if step.get("block") == "load_csv":
            step.setdefault("params", {})["filepath"] = str(tmp_csv)
            break

    # ── 6. Execute the pipeline ──────────────────────────────────────
    try:
        engine = PipelineEngine(pipeline)
        final_output, artifacts = engine.execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}")
    finally:
        # Clean up temp file
        if tmp_csv.exists():
            tmp_csv.unlink()

    # ── 7. Build response ────────────────────────────────────────────
    result_summary = {
        "status": "success",
        "steps_executed": len(pipeline),
        "final_output_type": type(final_output).__name__,
        "artifacts": artifacts,
    }

    # If the final output is a DataFrame, include a preview
    if isinstance(final_output, pd.DataFrame):
        result_summary["preview"] = json.loads(
            final_output.head(10).to_json(orient="records", date_format="iso")
        )

    # If the final output is a trained model, include basic info
    if hasattr(final_output, "get_params"):
        result_summary["model_params"] = final_output.get_params()
    if hasattr(final_output, "score"):
        result_summary["model_type"] = type(final_output).__name__

    return result_summary


# ── Chat ─────────────────────────────────────────────────────────────

@app.post("/api/chat", summary="Chat with Nexus Assistant")
async def chat_with_assistant(body: ChatRequest):
    """Send a conversation history to Ollama and get the assistant's next response."""
    messages_dict = [msg.model_dump() for msg in body.messages]
    
    try:
        response_text = generate_chat_response(messages_dict)
        return {"response": response_text}
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Chat orchestrator error: {exc}",
        )


# ── Metadata ─────────────────────────────────────────────────────────

@app.get("/api/metadata", summary="Get block metadata")
async def get_metadata():
    """Return the metadata.json registry so the UI knows allowed block parameters."""
    metadata_path = Path(__file__).parent / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Health check ─────────────────────────────────────────────────────

@app.get("/", summary="Health check")
async def health():
    return {"status": "ok", "service": "Astrikos API", "version": "1.0.0"}


# ── Run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)