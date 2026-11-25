"""HTTP prediction service for the selected model."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Predictive Maintenance API", version="0.1.0")


class MachineReading(BaseModel):
    type: Literal["L", "M", "H"]
    air_temperature: float = Field(gt=0)
    process_temperature: float = Field(gt=0)
    rotational_speed: int = Field(gt=0)
    torque: float = Field(ge=0)
    tool_wear: int = Field(ge=0)


@lru_cache(maxsize=1)
def load_model():
    path = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {path}. Run training first.")
    return joblib.load(path)


@app.get("/health")
def health():
    try:
        load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok"}


@app.post("/predict")
def predict(reading: MachineReading):
    try:
        bundle = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    frame = pd.DataFrame([{
        "Type": reading.type,
        "Air temperature [K]": reading.air_temperature,
        "Process temperature [K]": reading.process_temperature,
        "Rotational speed [rpm]": reading.rotational_speed,
        "Torque [Nm]": reading.torque,
        "Tool wear [min]": reading.tool_wear,
    }])
    probability = float(bundle["model"].predict_proba(frame)[:, 1][0])
    threshold = float(bundle["threshold"])
    return {"failure_probability": probability, "predicted_failure": probability >= threshold, "threshold": threshold}
