"""Download the original CSV from the UCI repository."""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .features import select_features

SOURCE = "https://archive.ics.uci.edu/static/public/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
DEFAULT_OUTPUT = Path("data/raw/ai4i2020.csv")


def download(output: Path = DEFAULT_OUTPUT) -> Path:
    output = Path(output)
    with urllib.request.urlopen(SOURCE, timeout=30) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        with zipped.open("ai4i2020.csv") as csv_file:
            content = csv_file.read()
    frame = pd.read_csv(io.BytesIO(content))
    select_features(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    print(f"Downloaded {len(frame)} rows to {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
