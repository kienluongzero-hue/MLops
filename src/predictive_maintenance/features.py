"""The prediction contract shared by training and serving."""

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NUMERIC_FEATURES = FEATURES[1:]
TARGET = "Machine failure"


def select_features(frame):
    """Use an explicit allowlist so failure mode labels cannot leak into X."""
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Required features and target must not contain nulls")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Machine failure must be binary")
    if not set(frame["Type"].unique()).issubset({"L", "M", "H"}):
        raise ValueError("Type must be L, M or H")
    return frame[FEATURES].copy(), frame[TARGET].astype(int).copy()
