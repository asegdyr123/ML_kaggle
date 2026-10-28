"""Export an explicitly specified final-model blend in original test-row order."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from common import ROOT, write_json


def blend_predictions(frames, weights, expected):
    weights = np.asarray(weights, dtype=float)
    if len(frames) != len(weights) or not len(frames):
        raise ValueError("One weight required for each prediction frame")
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Weights must be finite, nonnegative, and sum to one")
    prediction = np.zeros(len(expected), dtype=np.float64)
    for frame, weight in zip(frames, weights):
        frame = frame.sort("row_index")
        if frame["row_index"].n_unique() != len(frame):
            raise ValueError("Duplicate row_index")
        for column in ["row_index", "log_id", "pt_d"]:
            if not frame[column].equals(expected[column]):
                raise ValueError(f"Test identity mismatch: {column}")
        if "label" in frame.columns and not (frame["label"] == -1).all():
            raise ValueError("Expected hidden-test predictions only")
        p = frame["pctr"].to_numpy()
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid probabilities")
        prediction += weight * p
    return prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    recipe = json.loads(args.config.read_text(encoding="utf-8"))
    out = ROOT / "outputs" / recipe["name"]
    csv_path = out / "submission.csv"
    if csv_path.exists():
        raise FileExistsError(f"Preserve existing submission: {csv_path}")
    expected = (pl.scan_csv(ROOT / "2026ml/final_test.csv")
                .select("log_id", "pt_d").with_row_index("row_index").collect())
    assert expected["pt_d"].unique().to_list() == [20191117]
    train_rows = pl.scan_csv(ROOT / "2026ml/train.csv").select(pl.len()).collect().item()
    frames, evidence = [], []
    for name in recipe["members"]:
        directory = ROOT / "outputs" / name
        metadata = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
        if not metadata["args"]["final"] or metadata["validation_day"] != 20191117:
            raise ValueError(f"Not a final refit: {name}")
        if metadata["full_train_rows"] != train_rows:
            raise ValueError(f"Not all training dates included: {name}")
        path = directory / "test_predictions.parquet"
        frames.append(pl.read_parquet(path))
        evidence.append({"name": name, "iterations": metadata["best_iterations"],
                         "sampled_train_rows": metadata["train_rows"],
                         "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    prediction = blend_predictions(frames, recipe["weights"], expected)
    out.mkdir(parents=True, exist_ok=True)
    csv = pl.DataFrame({"id": np.arange(len(expected)), "pctr": prediction})
    temp = out / "submission.csv.tmp"
    csv.write_csv(temp, float_precision=10, include_bom=False)
    checked = pl.read_csv(temp)
    assert checked.columns == ["id", "pctr"]
    assert np.array_equal(checked["id"].to_numpy(), np.arange(len(expected)))
    assert np.allclose(checked["pctr"].to_numpy(), prediction, rtol=0, atol=5.1e-11)
    temp.rename(csv_path)
    expected.with_columns(pl.Series("pctr", prediction)).write_parquet(out / "test_predictions.parquet")
    write_json(out / "manifest.json", {
        "recipe": recipe, "members": evidence, "training_rows_before_sampling": train_rows,
        "test_rows": len(expected), "columns": checked.columns,
        "id_rule": "zero-based original final_test.csv row position; log_id is not unique",
        "format_evidence": "Captain's accepted 2026-10-25 submission package: id,pctr with zero-based row IDs",
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "minimum_pctr": float(prediction.min()), "maximum_pctr": float(prediction.max()),
        "mean_pctr": float(prediction.mean()),
        "iteration_rule": "Use selected day-16 best iteration counts unchanged; train on all dates, retaining configured negative sampling",
        "kaggle_score": None, "upload_status": "not_uploaded"})
    print(json.dumps({"csv": str(csv_path), "rows": len(expected), "sha256":
        hashlib.sha256(csv_path.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
