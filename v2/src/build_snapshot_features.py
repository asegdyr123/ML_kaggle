"""Within-day history changes, allowed by the classroom's feature policy.

Uses all provided advertisement feature rows for the SAME user/date only.
No advertisement labels and no subsequent-day data are read into features.
"""
import time
import polars as pl
from common import ROOT, write_json


if __name__ == "__main__":
    start = time.monotonic()
    seqs = [f"ad_{action}_list_v00{i}" for action in ("click", "close") for i in (1, 2, 3)]
    fields = ["user_id", "pt_d", "task_id", "adv_prim_id", "spread_app_id"] + seqs
    raw = pl.concat([pl.scan_csv(ROOT / "2026ml" / name, schema_overrides={c:pl.String for c in seqs})
                      .select(fields).with_row_index("row_index").with_columns(pl.lit(test).cast(pl.Int8).alias("is_test"))
                     for name, test in [("train.csv", 0), ("final_test.csv", 1)]]).collect(engine="streaming")
    result = raw.select(["row_index", "is_test"])
    keys = ["user_id", "pt_d"]
    for seq in seqs:
        target = {"1":"task_id", "2":"adv_prim_id", "3":"spread_app_id"}[seq[-1]]
        prefix = "snapshot_" + seq.replace("ad_", "").replace("_list_v00", "_")
        snapshots = raw.select(keys + [seq]).unique()
        totals = snapshots.group_by(keys).len().rename({"len":"_total"})
        tokens = snapshots.with_columns(pl.col(seq).str.split("^").alias("_token")).explode("_token").unique(keys + [seq, "_token"])
        counts = tokens.group_by(keys + ["_token"]).len().rename({"len":"_matches"})
        union = counts.group_by(keys).len().rename({"len":"_union_size"})
        selected = raw.select(keys + [target]).with_columns(pl.col(target).cast(pl.String).alias("_token"))
        selected = selected.join(counts, on=keys + ["_token"], how="left", maintain_order="left").join(totals, on=keys, how="left", maintain_order="left").join(union,on=keys,how="left",maintain_order="left")
        matches = pl.col("_matches").fill_null(0)
        new = selected.select(
            pl.col("_total").cast(pl.Float32).alias(prefix + "_count"),
            pl.col("_union_size").cast(pl.Float32).alias(prefix + "_union_size"),
            (matches > 0).cast(pl.Float32).alias(prefix + "_union_match"),
            ((matches > 0) & (matches < pl.col("_total"))).cast(pl.Float32).alias(prefix + "_membership_changed"),
            (matches / pl.col("_total")).cast(pl.Float32).alias(prefix + "_match_fraction"))
        assert len(new) == len(result)
        result = result.hstack(new)
        print(f"Built {prefix}", flush=True)
    output = ROOT / "cache" / "snapshot_rows.parquet"
    result.write_parquet(output, compression="zstd")
    write_json(output.with_suffix(".json"), {"features":result.columns[2:], "rows":len(result),
        "target_labels_used":False, "time_boundary":"same user and same calendar day only",
        "elapsed_seconds":time.monotonic()-start})
    print(f"Saved snapshot features in {time.monotonic()-start:.1f}s", flush=True)
