"""Run experiments sequentially and keep complete logs across chat turns."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import ROOT, write_json


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    args = p.parse_args()
    configs = json.loads(args.config.read_text(encoding="utf-8"))
    logs = ROOT / "outputs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    statuses = []
    for item in configs:
        name = item["name"]
        if (ROOT / "outputs" / name / "experiment.json").exists():
            print(f"SKIP completed {name}", flush=True)
            continue
        command = [sys.executable, "-u", "-X", "utf8", str(ROOT / "src" / "run_experiment.py"), "--name", name] + item["args"]
        print(f"START {name}", flush=True)
        with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        statuses.append({"name": name, "returncode": code})
        write_json(logs / f"{args.config.stem}_status.json", statuses)
        print(f"FINISH {name}: {code}", flush=True)
        if code:
            sys.exit(code)
