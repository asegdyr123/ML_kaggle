"""Check rate-only feature selection on real, synthetic LODO statistics; no training."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_experiment import add_target_stats, target_stat_features, to_pandas


def main():
    # Deliberately unsorted rows and an unseen validation key exercise join identity.
    frame = pl.DataFrame({
        "row_index": [40, 10, 30, 20, 50, 60],
        "pt_d": [20191112, 20191110, 20191111, 20191110, 20191111, 20191112],
        "label": [0, 1, 0, 0, 1, 1],
        "user_id": [1, 1, 1, 2, 2, 9],
        "slot_id": [2, 2, 2, 3, 3, 4],
        "click_1_sequence_id": [6, 2, 4, 3, 5, 7],
        "sequence_clicks": [9, 2, 3, 4, 5, 6],
        "source_n": [5, 4, 3, 2, 1, 0],
        "user_count": [2, 3, 4, 5, 6, 7],
    })
    cutoff = 20191112
    fit = frame.filter(pl.col("pt_d") < cutoff)
    encoded, names = add_target_stats(
        frame, fit, cutoff, "lodo", [["user_id"], ["user_id", "slot_id"]])
    raw = ["te_user_id_n", "te_user_id_clicks",
           "te_user_id_slot_id_n", "te_user_id_slot_id_clicks"]
    rates = ["te_user_id_ctr", "te_user_id_slot_id_ctr"]
    assert names == [raw[0], raw[1], rates[0], raw[2], raw[3], rates[1]]
    base = ["row_index", "click_1_sequence_id", "sequence_clicks", "source_n", "user_count"]
    default = to_pandas(encoded, base + target_stat_features(names))
    explicit_false = to_pandas(encoded, base + target_stat_features(names, False))
    rate_only = to_pandas(encoded, base + target_stat_features(names, True))
    pd.testing.assert_frame_equal(default, to_pandas(encoded, base + names))
    pd.testing.assert_frame_equal(default, explicit_false)
    assert list(rate_only.columns) == base + rates
    assert set(default.columns) - set(rate_only.columns) == set(raw)
    pd.testing.assert_frame_equal(rate_only, default[base + rates])
    assert encoded["row_index"].to_list() == frame["row_index"].to_list()
    assert rate_only["row_index"].tolist() == [40, 10, 30, 20, 50, 60]
    # Counts are still computed (including unseen-key zeros) even when excluded.
    np.testing.assert_array_equal(encoded[raw[0]].to_numpy(), [2, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(encoded[raw[1]].to_numpy(), [1, 0, 1, 1, 0, 0])
    expected_ctr = np.array([51 / 102, 50 / 101, 51 / 101, 51 / 101, 50 / 101, .5])
    np.testing.assert_allclose(rate_only[rates[0]], expected_ctr, rtol=1e-6)
    assert all(name in encoded.columns for name in raw + rates)
    print("PASS: default unchanged; rate-only matrices preserve rows, CTRs and unrelated features")


if __name__ == "__main__":
    main()
