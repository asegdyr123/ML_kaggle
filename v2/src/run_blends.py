"""Evaluate explicitly specified, reproducible blend recipes; never optimize weights."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from common import ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    for item in json.loads(args.config.read_text(encoding="utf-8")):
        command = [sys.executable, "-X", "utf8", str(ROOT / "src/evaluate_ensemble.py"),
                   "--name", item["name"], "--members", *item["members"]]
        if "weights" in item:
            command += ["--weights", *map(str, item["weights"])]
        if item.get("reference"):
            command += ["--reference", item["reference"]]
        if item.get("bootstrap"):
            command += ["--bootstrap", str(item["bootstrap"])]
        subprocess.run(command, cwd=ROOT, check=True)
