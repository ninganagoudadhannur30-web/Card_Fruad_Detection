"""Fraud scoring API.

Loads the trained XGBoost model and exposes it over HTTP.
Run from the project root with:  uvicorn app.main:app --reload
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI
from pydantic import BaseModel, Field, create_model
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "xgb_model.json"
CONFIG_PATH = ROOT / "model" / "config.json"
SAMPLE_PATH = ROOT / "sample" / "sample_transactions.json"

# --- Load the model and its settings once, when the server starts ---------
if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
    raise RuntimeError(
        "Model files not found. Run the export cell at the end of the notebook "
        "to create model/xgb_model.json and model/config.json."
    )

model = XGBClassifier()
model.load_model(MODEL_PATH)
config = json.loads(CONFIG_PATH.read_text())
THRESHOLD = config["threshold"]
FEATURES = config["features"]  # V1..V28 followed by log_amount

# --- Input and output shapes ----------------------------------------------
# V1..V28 are required floats, Amount must be >= 0.
fields = {f"V{i}": (float, ...) for i in range(1, 29)}
fields["Amount"] = (float, Field(..., ge=0))
Transaction = create_model("Transaction", **fields)


class Prediction(BaseModel):
    fraud_score: float
    flagged: bool
    threshold: float


# Demo transactions shown in the interactive docs page (if the file exists)
EXAMPLES = {}
if SAMPLE_PATH.exists():
    samples = json.loads(SAMPLE_PATH.read_text())
    EXAMPLES = {
        name: {"summary": name.replace("_", " "), "value": row}
        for name, row in samples.items()
    }

app = FastAPI(
    title="Credit Card Fraud Scoring API",
    description=(
        "Learning project. Scores a transaction with an XGBoost model trained on an "
        "anonymized dataset. Not for real-world use."
    ),
    version="1.0.0",
)


@app.get("/health")
def health():
    """Check that the service is running."""
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Fraud scoring API. See /docs for the interactive page."}

@app.get("/model-info")
def model_info():
    """Show the decision threshold and the features the model expects."""
    return {"model": "XGBoost", "threshold": THRESHOLD, "features": FEATURES}


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction = Body(..., openapi_examples=EXAMPLES)):
    """Return the fraud score for one transaction and whether it is flagged."""
    row = tx.model_dump()
    row["log_amount"] = float(np.log1p(row.pop("Amount")))  # same transform as training
    X = pd.DataFrame([row])[FEATURES]                        # same column order as training
    score = float(model.predict_proba(X)[0, 1])
    return Prediction(fraud_score=round(score, 4), flagged=score >= THRESHOLD, threshold=THRESHOLD)
