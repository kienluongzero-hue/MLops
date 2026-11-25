"""Make a deterministic, clearly labelled demo batch from held-out data."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .features import select_features

DEFAULT_DATA = Path("data/raw/ai4i2020.csv")
DEFAULT_OUTPUT = Path("data/monitoring/current.csv")
SEED = 42


def simulate(data_path=DEFAULT_DATA, output_path=DEFAULT_OUTPUT, rows=500, drift=False):
    frame = pd.read_csv(data_path)
    features, target = select_features(frame)
    _, held_out, _, _ = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=SEED
    )
    if rows < 1 or rows > len(held_out):
        raise ValueError(f"rows must be between 1 and {len(held_out)}")
    batch = held_out.sample(n=rows, random_state=SEED).reset_index(drop=True)
    if drift:
        batch["Air temperature [K]"] += 5.0
        batch["Torque [Nm]"] += 10.0
        batch.loc[: int(rows * 0.6) - 1, "Type"] = "H"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(output_path, index=False)
    print(f"Wrote {rows} {'shifted' if drift else 'unchanged'} demo rows to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--drift", action="store_true")
    args = parser.parse_args()
    simulate(args.data, args.output, args.rows, args.drift)


if __name__ == "__main__":
    main()
