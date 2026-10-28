"""Label-independent same-day features for the classroom's allowed transductive setting.

No target-domain labels participate in these transformations. Source-domain labels
are supplied inputs (including test_feeds.csv), not hidden target labels.
"""
import argparse
import gc
import time
from pathlib import Path

import polars as pl

from common import ROOT, write_json

SEQUENCES = [f"ad_{action}_list_v00{i}" for action in ("click", "close") for i in (1, 2, 3)] + ["u_newsCatInterestsST"]
CAT_ORIGINAL = ["user_id", "age", "sex", "residence", "city", "city_rank", "series_dev", "series_group",
                "device_dev", "device_name", "device_size", "net_type", "task_id", "adv_id", "creat_type_cd",
                "adv_prim_id", "inter_type_cd", "slot_id", "site_id", "spread_app_id", "hispace_app_tags",
                "app_second_class", "u_refreshTimes", "u_feedLifeCycle"]


def load_ads(data):
    overrides = {name: pl.String for name in SEQUENCES}
    train = pl.read_csv(data / "train.csv", schema_overrides=overrides).with_row_index("row_index").with_columns(pl.lit(0).alias("is_test"))
    test = pl.read_csv(data / "final_test.csv", schema_overrides=overrides).with_row_index("row_index").with_columns(pl.lit(-1).cast(pl.Int64).alias("label"), pl.lit(1).alias("is_test"))
    return pl.concat([train, test.select(train.columns)], how="vertical_relaxed")


def source_daily(data):
    print("Aggregating source-domain user/day activity", flush=True)
    numeric = ["u_phonePrice", "u_browserLifeCycle", "u_browserMode", "u_feedLifeCycle", "u_refreshTimes",
               "i_dislikeTimes", "i_upTimes", "e_po", "e_rn", "pro"]
    categorical = ["i_docId", "i_s_sourceId", "i_cat", "e_ch", "e_m", "e_pl", "i_dtype"]
    selected = ["u_userId", "e_et", "label", "cillabel"] + numeric + categorical
    source = pl.concat([pl.scan_csv(data / name, infer_schema_length=10000).select(selected)
                        for name in ("train_feeds.csv", "test_feeds.csv")])
    aggs = [pl.len().alias("src_exposures"), (pl.col("label") == 1).sum().alias("src_clicks"),
            (pl.col("label") == 1).mean().alias("src_ctr"),
            (pl.col("cillabel") == 1).sum().alias("src_likes")]
    for c in numeric:
        aggs += [pl.col(c).mean().alias(f"src_{c}_mean"), pl.col(c).max().alias(f"src_{c}_max")]
    for c in categorical:
        aggs += [pl.col(c).n_unique().alias(f"src_{c}_unique")]
    for c in ["u_phonePrice", "u_browserLifeCycle", "u_browserMode", "e_m", "e_pl"]:
        aggs.append(pl.col(c).mode().first().alias(f"src_{c}_mode"))
    # Fixed semantic IDs, selected without target labels. Coverage is measured by frequency only.
    top_cats = source.group_by("i_cat").len().sort("len", descending=True).head(20).collect()["i_cat"].to_list()
    for cat in top_cats:
        aggs.append((pl.col("i_cat") == cat).mean().alias(f"src_topic_{cat}_share"))
        aggs.append(((pl.col("i_cat") == cat) & (pl.col("label") == 1)).mean().alias(f"src_topic_{cat}_click_share"))
    return source.group_by(["u_userId", "e_et"]).agg(aggs).collect(engine="streaming").rename({"u_userId": "user_id", "e_et": "pt_d"})


def build(data, output):
    started = time.monotonic()
    frame = load_ads(data)
    print(f"Loaded {len(frame):,} ad rows", flush=True)
    groups = {"base": [c for c in frame.columns if c not in SEQUENCES + ["label", "row_index", "is_test"]],
              "sequence": [], "counts": [], "source": []}
    cats = CAT_ORIGINAL.copy()
    exprs = []
    for col in SEQUENCES:
        prefix = col.replace("ad_", "").replace("_list_v00", "_").replace("u_newsCatInterestsST", "news_st")
        tokens = pl.col(col).fill_null("").str.split("^")
        seq_exprs = [tokens.list.len().alias(prefix + "_len"),
                     tokens.list.n_unique().alias(prefix + "_unique"),
                     pl.col(col).cast(pl.Categorical).to_physical().cast(pl.Int32).alias(prefix + "_sequence_id")]
        cats.append(prefix + "_sequence_id")
        for i in range(5):
            name = f"{prefix}_token{i}"
            seq_exprs.append(tokens.list.get(i, null_on_oob=True).cast(pl.Int32, strict=False).fill_null(-1).alias(name))
            cats.append(name)
        if col.startswith("ad_"):
            target = {"1": "task_id", "2": "adv_prim_id", "3": "spread_app_id"}[col[-1]]
            seq_exprs.append(tokens.list.contains(pl.col(target).cast(pl.String)).alias(prefix + "_match"))
            seq_exprs.append((tokens.list.first() == pl.col(target).cast(pl.String)).alias(prefix + "_first_match"))
        groups["sequence"].extend(e.meta.output_name() for e in seq_exprs)
        exprs.extend(seq_exprs)
    frame = frame.with_columns(exprs).drop(SEQUENCES)
    print("Sequence features ready", flush=True)
    exprs = []
    count_keys = [["user_id"], ["task_id"], ["adv_id"], ["adv_prim_id"], ["spread_app_id"], ["slot_id"],
                  ["user_id", "task_id"], ["user_id", "adv_prim_id"], ["user_id", "spread_app_id"],
                  ["user_id", "slot_id"], ["task_id", "slot_id"]]
    for keys in count_keys:
        name = "day_count_" + "_".join(keys)
        exprs.append(pl.len().over(keys + ["pt_d"]).alias(name))
    for c in ["task_id", "adv_prim_id", "spread_app_id", "slot_id", "net_type", "device_size"]:
        exprs.append(pl.col(c).n_unique().over(["user_id", "pt_d"]).alias("user_day_unique_" + c))
    exprs.extend([
        (pl.col("log_id") / pl.col("log_id").max().over("pt_d")).alias("log_id_day_fraction"),
        ((pl.col("log_id") - pl.col("log_id").min().over(["user_id", "pt_d"])) /
         (pl.col("log_id").max().over(["user_id", "pt_d"]) - pl.col("log_id").min().over(["user_id", "pt_d"]) + 1)).alias("log_id_user_day_fraction"),
    ])
    groups["counts"] = [e.meta.output_name() for e in exprs]
    frame = frame.with_columns(exprs)
    print("Unlabelled exposure counts ready", flush=True)
    src = source_daily(data)
    groups["source"] = [c for c in src.columns if c not in ["user_id", "pt_d"]]
    cats.extend(c for c in groups["source"] if c.endswith("_mode"))
    frame = frame.join(src, on=["user_id", "pt_d"], how="left", maintain_order="left")
    del src
    # Contiguous category IDs avoid large raw categorical codes in LightGBM.
    # Their vocabulary uses unlabelled feature values only.
    frame = frame.with_columns([pl.col(c).fill_null(-1).cast(pl.String).cast(pl.Categorical)
                                .to_physical().cast(pl.Int32).alias(c) for c in cats])
    frame = frame.with_columns([pl.col(c).cast(pl.Float32).fill_null(-1).alias(c)
                                for c in frame.columns if c not in cats + ["row_index", "is_test", "label", "pt_d", "log_id"]])
    frame = frame.with_columns(pl.col("label").cast(pl.Int8), pl.col("is_test").cast(pl.Int8), pl.col("pt_d").cast(pl.Int32), pl.col("log_id").cast(pl.Int32))
    # Same-date joins must not expand or reorder advertisement rows.
    assert len(frame) == frame.filter(pl.col("is_test") == 0)["row_index"].n_unique() + frame.filter(pl.col("is_test") == 1)["row_index"].n_unique()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output, compression="zstd")
    write_json(output.with_suffix(".json"), {"groups": groups, "categorical": cats,
        "rows": len(frame), "columns": len(frame.columns), "source": str(data.resolve()),
        "target_labels_used_for_features": False, "source_join": "user_id + same calendar day",
        "elapsed_seconds": time.monotonic() - started})
    print(f"Saved {frame.shape} to {output}; elapsed {time.monotonic()-started:.1f}s", flush=True)
    del frame
    gc.collect()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "2026ml")
    p.add_argument("--output", type=Path, default=ROOT / "cache" / "features_v1.parquet")
    args = p.parse_args()
    build(args.data, args.output)
