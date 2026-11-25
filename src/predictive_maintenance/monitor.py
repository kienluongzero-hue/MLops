"""Compare an incoming feature batch with the training reference."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .features import FEATURES, NUMERIC_FEATURES

DEFAULT_REFERENCE = Path("artifacts/reference.csv")
DEFAULT_OUTPUT = Path("artifacts/monitoring/report.json")
CATEGORIES = ("L", "M", "H")
MIN_ROWS = 100
KS_THRESHOLD = 0.20
P_VALUE_THRESHOLD = 0.01
TV_THRESHOLD = 0.10


def _read_features(path):
    frame = pd.read_csv(path)
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    if frame.empty:
        raise ValueError(f"No rows in {path}")
    return frame[FEATURES]


def _valid_numeric(series):
    values = pd.to_numeric(series, errors="coerce")
    valid = values.notna() & np.isfinite(values)
    return values[valid].to_numpy(dtype=float), int((~valid).sum())


def compare(reference_path, current_path, output_path=DEFAULT_OUTPUT):
    reference = _read_features(reference_path)
    current = _read_features(current_path)
    features = {}
    quality_issues = {}
    drifted = []

    reference_types = reference["Type"]
    if not reference_types.isin(CATEGORIES).all():
        raise ValueError("Reference contains invalid Type values")
    current_types = current["Type"]
    invalid_types = int((~current_types.isin(CATEGORIES)).sum())
    if invalid_types:
        quality_issues["Type"] = invalid_types
    reference_frequencies = reference_types.value_counts(normalize=True)
    current_frequencies = current_types[current_types.isin(CATEGORIES)].value_counts(normalize=True)
    if current_frequencies.empty:
        tv_distance = None
        type_drift = False
    else:
        tv_distance = float(sum(
            abs(reference_frequencies.get(category, 0.0) - current_frequencies.get(category, 0.0))
            for category in CATEGORIES
        ) / 2)
        type_drift = tv_distance >= TV_THRESHOLD
    features["Type"] = {
        "method": "total_variation_distance",
        "statistic": tv_distance,
        "threshold": TV_THRESHOLD,
        "invalid_rows": invalid_types,
        "drift_detected": type_drift,
    }
    if type_drift:
        drifted.append("Type")

    for name in NUMERIC_FEATURES:
        baseline, bad_reference = _valid_numeric(reference[name])
        if bad_reference:
            raise ValueError(f"Reference contains invalid values in {name}")
        observed, invalid = _valid_numeric(current[name])
        if invalid:
            quality_issues[name] = invalid
        if len(observed) == 0:
            statistic = p_value = None
            feature_drift = False
            current_mean = None
        else:
            test = ks_2samp(baseline, observed, method="auto")
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
            feature_drift = statistic >= KS_THRESHOLD and p_value < P_VALUE_THRESHOLD
            current_mean = float(observed.mean())
        features[name] = {
            "method": "kolmogorov_smirnov",
            "statistic": statistic,
            "p_value": p_value,
            "threshold": KS_THRESHOLD,
            "p_value_threshold": P_VALUE_THRESHOLD,
            "reference_mean": float(baseline.mean()),
            "current_mean": current_mean,
            "invalid_rows": invalid,
            "drift_detected": feature_drift,
        }
        if feature_drift:
            drifted.append(name)

    if len(current) < MIN_ROWS:
        quality_issues["minimum_batch_rows"] = MIN_ROWS - len(current)
    report = {
        "reference_rows": len(reference),
        "current_rows": len(current),
        "drift_detected": bool(drifted),
        "drifted_features": drifted,
        "data_quality_alert": bool(quality_issues),
        "quality_issues": quality_issues,
        "status": "alert" if drifted or quality_issues else "ok",
        "features": features,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "current_rows", "drifted_features", "quality_issues")}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    compare(args.reference, args.current, args.output)


if __name__ == "__main__":
    main()
