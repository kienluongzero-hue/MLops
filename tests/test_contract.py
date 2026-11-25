import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient

from predictive_maintenance.api import app, load_model
from predictive_maintenance.features import FEATURES, select_features


def test_failure_mode_labels_never_become_features():
    frame = pd.DataFrame([{
        "Type": "L", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 100,
        "Machine failure": 1, "TWF": 1, "HDF": 1, "PWF": 1, "OSF": 1, "RNF": 1,
    }])
    x, y = select_features(frame)
    assert list(x.columns) == FEATURES
    assert y.tolist() == [1]


def test_missing_feature_fails_before_training():
    with pytest.raises(ValueError, match="Missing required columns"):
        select_features(pd.DataFrame({"Machine failure": [0]}))


def test_api_returns_probability_and_decision(monkeypatch):
    class FakeModel:
        def predict_proba(self, frame):
            assert list(frame.columns) == FEATURES
            return np.array([[0.2, 0.8]])

    monkeypatch.setattr("predictive_maintenance.api.load_model", lambda: {"model": FakeModel(), "threshold": 0.6})
    response = TestClient(app).post("/predict", json={
        "type": "L", "air_temperature": 298.1, "process_temperature": 308.6,
        "rotational_speed": 1551, "torque": 42.8, "tool_wear": 0,
    })
    assert response.status_code == 200
    assert response.json() == {"failure_probability": 0.8, "predicted_failure": True, "threshold": 0.6}


def test_health_reports_missing_artifact(monkeypatch):
    load_model.cache_clear()
    monkeypatch.setenv("MODEL_PATH", "missing-model.joblib")
    response = TestClient(app).get("/health")
    assert response.status_code == 503
    load_model.cache_clear()
