import numpy as np
import pandas as pd

from predictive_maintenance.monitor import compare


def _reference():
    rng = np.random.default_rng(42)
    rows = 300
    return pd.DataFrame({
        "Type": rng.choice(["L", "M", "H"], size=rows, p=[0.5, 0.3, 0.2]),
        "Air temperature [K]": rng.normal(300, 2, rows),
        "Process temperature [K]": rng.normal(310, 2, rows),
        "Rotational speed [rpm]": rng.normal(1500, 100, rows),
        "Torque [Nm]": rng.normal(40, 10, rows),
        "Tool wear [min]": rng.integers(0, 240, rows),
    })


def _compare(tmp_path, reference, current):
    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"
    output_path = tmp_path / "report.json"
    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)
    report = compare(reference_path, current_path, output_path)
    assert output_path.is_file()
    return report


def test_same_distribution_has_no_alert(tmp_path):
    reference = _reference()
    report = _compare(tmp_path, reference, reference.copy())
    assert report["status"] == "ok"
    assert report["drifted_features"] == []


def test_shifted_batch_identifies_changed_features(tmp_path):
    reference = _reference()
    current = reference.copy()
    current["Air temperature [K]"] += 5
    current["Type"] = "H"
    report = _compare(tmp_path, reference, current)
    assert report["status"] == "alert"
    assert "Air temperature [K]" in report["drifted_features"]
    assert "Type" in report["drifted_features"]


def test_invalid_values_raise_quality_alert(tmp_path):
    reference = _reference()
    current = reference.copy()
    current.loc[0, "Type"] = "unknown"
    current.loc[1, "Torque [Nm]"] = np.nan
    report = _compare(tmp_path, reference, current)
    assert report["data_quality_alert"]
    assert report["quality_issues"] == {"Type": 1, "Torque [Nm]": 1}
