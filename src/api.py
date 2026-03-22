"""
Module: api.py
--------------
Role: Expose the trained model as a FastAPI HTTP service.
Responsibility: Deserialize requests, call serving modules, serialize responses.

Contract:
  - NO new ML logic lives here.
  - api.py only calls run_inference() from infer.py and load_model_for_serving()
    from utils.py. All feature engineering and model logic stays in src/.
  - Pydantic enforces the full JSON request contract at the API boundary.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import logging
from contextlib import asynccontextmanager
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Local ─────────────────────────────────────────────────────────────────────
from src.infer import run_inference
from src.utils import load_model_for_serving, read_config, setup_logger

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
CONFIG = read_config("config.yaml")
logger = logging.getLogger("mlops")

# Model is loaded once at startup and reused for every request
_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the server starts; release on shutdown."""
    global _model
    setup_logger(CONFIG["logging"]["log_file"], CONFIG["logging"]["level"])
    logger.info("[api] Starting up — loading model for serving.")
    _model = load_model_for_serving(CONFIG)
    logger.info("[api] Model loaded successfully.")
    yield
    logger.info("[api] Shutting down.")


app = FastAPI(
    title="Classification API",
    description="Serves predictions from the trained classification pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas (Pydantic enforces the data contract) ──────────

class PredictRequest(BaseModel):
    """
    Expected JSON payload for /predict.
    All fields are required. Types and constraints are enforced by Pydantic
    before any prediction logic runs — invalid payloads return HTTP 422.
    """
    Pclass: int = Field(..., description="Passenger class (1, 2, or 3)")
    Sex: str = Field(..., description="Passenger sex ('male' or 'female')")
    Age: float = Field(..., ge=0, description="Passenger age in years")
    SibSp: int = Field(..., ge=0, description="Number of siblings / spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents / children aboard")
    Fare: float = Field(..., ge=0, description="Ticket fare paid")
    Embarked: str = Field(..., description="Port of embarkation: C, Q, or S")


class PredictResponse(BaseModel):
    """Structured prediction output returned by /predict."""
    prediction: int
    survival_probability: Optional[float] = None
    outcome: Optional[str] = None
    high_confidence: Optional[bool] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", status_code=200)
def health():
    """
    Liveness check.
    Returns 200 when the service is running and the model is loaded.
    """
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse, status_code=200)
def predict(request: PredictRequest):
    """
    Run a single prediction.

    Pydantic validates the incoming JSON before this function runs.
    The validated data is converted to a DataFrame and passed directly
    to run_inference() — no ML logic is added here.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        X = pd.DataFrame([request.model_dump()])
        logger.info("[api] Received prediction request: %s", request.model_dump())

        result = run_inference(_model, X)
        row = result.iloc[0]

        response = PredictResponse(
            prediction=int(row["prediction"]),
            survival_probability=(
                float(row["survival_probability"])
                if "survival_probability" in row.index else None
            ),
            outcome=(
                str(row["outcome"])
                if "outcome" in row.index else None
            ),
            high_confidence=(
                bool(row["high_confidence"])
                if "high_confidence" in row.index else None
            ),
        )

        logger.info("[api] Prediction returned: %s", response.model_dump())
        return response

    except Exception as e:
        logger.error("[api] Prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))