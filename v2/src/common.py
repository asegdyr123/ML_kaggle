"""Project paths and JSON helpers. No import-time data or authentication access."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
EXPECTED = ("train.csv", "train_feeds.csv", "final_test.csv", "test_feeds.csv")


def find_data(directory=DATA):
    directory = Path(directory)
    result = {}
    for name in EXPECTED:
        candidates = list(directory.rglob(name)) if directory.exists() else []
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {name} under {directory}; found {len(candidates)}"
            )
        result[name] = candidates[0]
    return result


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temp, path)

