"""Temporal experiments, label-safe target encoding, and final refits."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from common import ROOT, write_json

TE_KEYS = [["user_id"], ["task_id"], ["adv_id"], ["adv_prim_id"], ["spread_app_id"],
           ["slot_id"], ["app_second_class"], ["user_id", "adv_prim_id"],
           ["user_id", "spread_app_id"], ["user_id", "slot_id"], ["user_id", "task_id"],
           ["task_id", "slot_id"], ["age", "spread_app_id"], ["sex", "adv_prim_id"]]
EXTRA_TE_KEYS = [["click_1_sequence_id"], ["click_2_sequence_id"], ["news_st_sequence_id"],
                 ["user_id", "inter_type_cd"], ["user_id", "creat_type_cd"],
                 ["spread_app_id", "slot_id"], ["user_id", "app_second_class"],
                 ["click_1_sequence_id", "slot_id"]]


def add_past_target_stats(frame, fit_labels, cutoff, encoding_keys=None):
    """Strictly historical counts and prior, using one cumulative row per key/day."""
    assert fit_labels["pt_d"].max() is None or fit_labels["pt_d"].max() < cutoff, "Validation/test labels entered target statistics"
    names = []
    # A positional index preserves duplicate IDs and arbitrary original row order.
    frame = frame.with_row_index("_te_order").sort("pt_d")
    daily = fit_labels.group_by("pt_d").agg(
        pl.len().cast(pl.Int64).alias("_past_n"),
        pl.col("label").cast(pl.Int64).sum().alias("_past_s"))
    daily = daily.sort("pt_d").with_columns(pl.col("_past_n", "_past_s").cum_sum())
    frame = frame.join_asof(daily, on="pt_d", strategy="backward", allow_exact_matches=False).with_columns(
        pl.when(pl.col("_past_n").fill_null(0) > 0)
        .then(pl.col("_past_s") / pl.col("_past_n")).otherwise(0.01).alias("_prior")
    ).drop(["_past_n", "_past_s"])
    for keys in (TE_KEYS if encoding_keys is None else encoding_keys):
        prefix = "te_" + "_".join(keys)
        daily = fit_labels.group_by(keys + ["pt_d"]).agg(
            pl.len().cast(pl.Int64).alias("_n"),
            pl.col("label").cast(pl.Int64).sum().alias("_s"))
        daily = daily.sort("pt_d").with_columns(pl.col("_n", "_s").cum_sum().over(keys))
        frame = frame.join_asof(daily, on="pt_d", by=keys, strategy="backward",
                                allow_exact_matches=False, check_sortedness=False)
        n, s = pl.col("_n").fill_null(0), pl.col("_s").fill_null(0)
        strength = 30 if len(keys) > 1 else 100
        frame = frame.with_columns(
            n.cast(pl.Float32).alias(prefix + "_n"),
            s.cast(pl.Float32).alias(prefix + "_clicks"),
            ((s + strength * pl.col("_prior")) / (n + strength)).cast(pl.Float32).alias(prefix + "_ctr")
        ).drop(["_n", "_s"])
        names += [prefix + suffix for suffix in ("_n", "_clicks", "_ctr")]
    return frame.sort("_te_order").drop(["_te_order", "_prior"]), names


def add_target_stats(frame, fit_labels, cutoff, mode, encoding_keys=None):
    """All validation/test labels are absent from fit_labels.

    LODO excludes the complete held-out training day, including the current row.
    This uses future *training* dates, explicitly allowed by classroom rules.
    Past mode uses strictly earlier dates for both group statistics and prior.
    """
    if mode == "past":
        return add_past_target_stats(frame, fit_labels, cutoff, encoding_keys)
    names = []
    prior = float(fit_labels["label"].mean())
    assert fit_labels["pt_d"].max() < cutoff, "Validation/test labels entered target statistics"
    day_totals = fit_labels.group_by("pt_d").agg(pl.len().alias("_day_n"), pl.col("label").cast(pl.Int32).sum().alias("_day_s"))
    total_n, total_s = len(fit_labels), int(fit_labels["label"].sum())
    frame = frame.join(day_totals, on="pt_d", how="left", maintain_order="left").with_columns(
        pl.when(pl.col("pt_d") < cutoff)
        .then((total_s - pl.col("_day_s")) / (total_n - pl.col("_day_n")))
        .otherwise(prior).alias("_prior"))
    if mode == "lodo":
        for keys in (TE_KEYS if encoding_keys is None else encoding_keys):
            prefix = "te_" + "_".join(keys)
            all_stats = fit_labels.group_by(keys).agg(pl.len().cast(pl.Int32).alias("_n"), pl.col("label").cast(pl.Int32).sum().alias("_s"))
            daily = fit_labels.group_by(keys + ["pt_d"]).agg(pl.len().cast(pl.Int32).alias("_dn"), pl.col("label").cast(pl.Int32).sum().alias("_ds"))
            frame = frame.join(all_stats, on=keys, how="left", maintain_order="left").join(daily, on=keys + ["pt_d"], how="left", maintain_order="left")
            n = pl.col("_n").fill_null(0) - pl.when(pl.col("pt_d") < cutoff).then(pl.col("_dn").fill_null(0)).otherwise(0)
            s = pl.col("_s").fill_null(0) - pl.when(pl.col("pt_d") < cutoff).then(pl.col("_ds").fill_null(0)).otherwise(0)
            strength = 30 if len(keys) > 1 else 100
            frame = frame.with_columns(n.cast(pl.Float32).alias(prefix + "_n"), s.cast(pl.Float32).alias(prefix + "_clicks"),
                ((s + strength * pl.col("_prior")) / (n + strength)).cast(pl.Float32).alias(prefix + "_ctr"))
            frame = frame.drop(["_n", "_s", "_dn", "_ds"])
            names += [prefix + suffix for suffix in ("_n", "_clicks", "_ctr")]
    else:
        raise ValueError(mode)
    return frame.drop(["_day_n", "_day_s", "_prior"]), names


def corrected_probability(p, negative_fraction, positive_weight=1.0):
    # Weighted sampled odds = population odds * positive_weight / negative_fraction.
    # Apply this inverse exactly once, for both validation and final-test predictions.
    effective_fraction = negative_fraction / positive_weight
    return effective_fraction * p / (1 - p + effective_fraction * p)


def validate_positive_weight(model, negative_fraction, positive_weight):
    if not np.isfinite(negative_fraction) or not 0 < negative_fraction <= 1:
        raise ValueError("negative-fraction must be finite and in (0, 1]")
    if not np.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("positive-weight must be finite and positive")
    if positive_weight != 1 and model != "catboost":
        raise ValueError("positive-weight other than 1 is supported only for CatBoost")
    return negative_fraction / positive_weight


def catboost_class_weight_params(positive_weight):
    # Omit the parameter at the default to preserve existing CatBoost behavior.
    return {} if positive_weight == 1 else {"class_weights": [1.0, positive_weight],
                                           "eval_metric": "AUC:use_weights=false"}


def target_stat_features(names, rates_only=False):
    """Select model columns without changing the computed target statistics."""
    return [name for name in names if not rates_only or name.endswith("_ctr")]


def to_pandas(frame, columns):
    # Avoid a full homogeneous float64 matrix or object conversion.
    return pd.DataFrame({c: frame[c].to_numpy() for c in columns}, copy=False)


def specialist_train_filter(frame, column=None, value=None):
    """Select training rows only; values refer to the supplied parquet cache."""
    if (column is None) != (value is None):
        raise ValueError("train-group-column and train-group-value must be specified together")
    if column is None:
        return frame
    if column not in frame.columns:
        raise ValueError(f"Unknown train-group-column: {column}")
    selected = frame.filter(pl.col(column) == value)
    if not len(selected):
        raise ValueError(f"No training rows for {column}={value}")
    return selected


def run(args):
    positive_weight = getattr(args, "positive_weight", 1.0)
    effective_fraction = validate_positive_weight(args.model, args.negative_fraction, positive_weight)
    group_column = getattr(args, "train_group_column", None)
    group_value = getattr(args, "train_group_value", None)
    if (group_column is None) != (group_value is None):
        raise ValueError("train-group-column and train-group-value must be specified together")
    pairwise = getattr(args, "xgb_objective", "logloss") == "pairwise"
    if pairwise and (args.model != "xgboost" or args.half_life):
        raise ValueError("Pairwise objective requires xgboost and does not support half-life/sample weights")
    start = time.monotonic()
    iteration_metadata = {}
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = ROOT / "outputs" / args.name
    if output.exists() and (output / "experiment.json").exists():
        raise FileExistsError(f"Experiment already completed: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "running.json", {"args": vars(args), "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    metadata = json.loads(args.features.with_suffix(".json").read_text(encoding="utf-8"))
    selected_groups = args.groups.split(",")
    features = list(dict.fromkeys(c for group in selected_groups for c in metadata["groups"][group]))
    features = [c for c in features if c not in args.drop.split(",") and c != "pt_d"]
    if "day_number" not in args.drop.split(","):
        features += ["day_number"]
    encoding_keys = TE_KEYS + (EXTRA_TE_KEYS if args.extended_te else [])
    keys = list(dict.fromkeys(c for cols in encoding_keys for c in cols)) if args.target_encoding else []
    if group_column is not None and group_column not in pl.read_parquet_schema(args.features):
        raise ValueError(f"Unknown train-group-column: {group_column}")
    cols = list(dict.fromkeys([c for c in features if c != "day_number"] + keys +
                             ([group_column] if group_column is not None else []) +
                             ["pt_d", "label", "row_index", "is_test", "log_id"]))
    data = pl.read_parquet(args.features, columns=cols).filter((pl.col("pt_d") <= args.day) & (pl.col("pt_d") >= args.train_start_day)).with_columns(
        (pl.col("pt_d") - 20191110).cast(pl.Float32).alias("day_number"))
    fit_labels = data.filter((pl.col("pt_d") < args.day) & (pl.col("is_test") == 0)).select(list(dict.fromkeys(keys + ["pt_d", "label"])))
    full_train_count = len(fit_labels)
    specialist_full_train_count = (len(specialist_train_filter(
        data.filter((pl.col("pt_d") < args.day) & (pl.col("is_test") == 0)),
        group_column, group_value)) if group_column is not None else None)
    train_mask = ((data["pt_d"].to_numpy() < args.day) & (data["is_test"].to_numpy() == 0))
    valid_mask = data["pt_d"].to_numpy() == args.day
    rng = np.random.default_rng(args.seed)
    sample_mask = (data["label"].to_numpy() == 1) | (rng.random(len(data)) < args.negative_fraction)
    data = data.filter(pl.Series((train_mask & sample_mask) | valid_mask))
    if args.interest:
        interest = pl.read_parquet(args.interest)
        interest_cols = [c for c in interest.columns if c not in ["row_index", "is_test"]]
        data = data.join(interest, on=["row_index", "is_test"], how="left", maintain_order="left", validate="1:1")
        features += interest_cols
        del interest
    for path in args.extra:
        extra = pl.read_parquet(path)
        extra_cols = [c for c in extra.columns if c not in ["row_index", "is_test"]]
        if set(extra_cols) & set(features):
            raise ValueError("Extra feature names overlap existing features")
        data = data.join(extra, on=["row_index", "is_test"], how="left", maintain_order="left", validate="1:1")
        features += extra_cols
        del extra
    if args.target_encoding:
        print("Building target statistics using training dates only", flush=True)
        data, names = add_target_stats(data, fit_labels, args.day, args.target_encoding, encoding_keys)
        features += target_stat_features(names, args.te_rates_only)
    del fit_labels
    gc.collect()
    # Drop constants using training data; the column vocabulary never uses target labels.
    train = specialist_train_filter(data.filter(pl.col("pt_d") < args.day), group_column, group_value)
    valid = data.filter(pl.col("pt_d") == args.day)
    evaluation_group_mask = (valid[group_column].eq(group_value).fill_null(False).to_numpy()
                             if group_column is not None else None)
    del data
    unique = train.select([pl.col(c).n_unique().alias(c) for c in features]).row(0, named=True)
    features = [c for c in features if unique[c] > 1]
    cats = [c for c in metadata["categorical"] if c in features]
    if args.cat_mode == "numeric":
        cats = []
    elif args.cat_mode == "lowcard":
        cats = [c for c in cats if unique[c] <= 512]
    y_train = train["label"].to_numpy()
    y_valid = valid["label"].to_numpy()
    weights = None
    if args.half_life:
        age_days = (args.day - 1 - train["pt_d"].to_numpy()).astype("float32")
        weights = np.power(0.5, age_days / args.half_life).astype("float32")
        weights /= weights.mean()
    identifiers = valid.select(["row_index", "log_id", "label", "pt_d"])
    x_train = to_pandas(train, features)
    x_valid = to_pandas(valid, features)
    del train, valid
    gc.collect()
    print(f"{args.name}: fit {len(x_train):,}/{full_train_count:,}, evaluate {len(x_valid):,}; features {len(features)}, categorical {len(cats)}", flush=True)
    if args.model in ["lightgbm", "xgboost"]:
        category_mapping = {}
        for c in cats:
            levels = np.unique(x_train[c].to_numpy())
            train_values = x_train[c].to_numpy()
            valid_values = x_valid[c].to_numpy()
            x_train[c] = np.searchsorted(levels, train_values).astype("int32")
            locations = np.searchsorted(levels, valid_values)
            seen = (locations < len(levels)) & (levels[np.minimum(locations, len(levels)-1)] == valid_values)
            x_valid[c] = np.where(seen, locations, -1).astype("int32")
            category_mapping[c] = levels.tolist()
        write_json(output / "category_mapping.json", category_mapping)
    if args.model == "lightgbm":
        import lightgbm as lgb
        # A contiguous NumPy matrix avoids pandas categorical copies.
        params = dict(objective="binary", metric="auc", n_estimators=args.iterations,
            learning_rate=args.learning_rate, num_leaves=args.leaves, max_depth=-1,
            min_child_samples=args.min_child, reg_lambda=args.l2, reg_alpha=args.l1,
            colsample_bytree=args.colsample, subsample=0.85, subsample_freq=1,
            max_bin=args.bins, max_cat_to_onehot=8, cat_smooth=args.cat_smooth,
            cat_l2=10, verbosity=-1, random_state=args.seed, n_jobs=args.threads,
            force_col_wise=True, deterministic=True)
        model = lgb.LGBMClassifier(**params)
        if args.final or args.fixed_iterations:
            model.fit(x_train, y_train, categorical_feature=cats, sample_weight=weights)
            best = args.iterations
        else:
            model.fit(x_train, y_train, categorical_feature=cats, sample_weight=weights, eval_set=[(x_valid, y_valid)],
                eval_metric="auc", callbacks=[lgb.early_stopping(args.patience, first_metric_only=True), lgb.log_evaluation(100)])
            best = model.best_iteration_
        prediction = model.predict_proba(x_valid)[:, 1]
        model.booster_.save_model(str(output / "model.txt"))
        importance = model.feature_importances_
    elif args.model == "xgboost":
        import xgboost as xgb
        feature_types = ["c" if c in cats else "q" for c in features]
        train_pool = xgb.QuantileDMatrix(x_train, label=y_train, weight=weights, feature_types=feature_types,
            enable_categorical=True, max_bin=args.bins, missing=-1, nthread=args.threads)
        valid_pool = xgb.QuantileDMatrix(x_valid, label=None if args.final else y_valid, ref=train_pool,
            feature_types=feature_types, enable_categorical=True, max_bin=args.bins, missing=-1, nthread=args.threads)
        del x_train, x_valid
        gc.collect()
        params = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", device="cuda",
            max_depth=args.depth, eta=args.learning_rate, min_child_weight=10, reg_lambda=args.l2,
            reg_alpha=args.l1, subsample=0.85, colsample_bytree=args.colsample,
            max_cat_to_onehot=8, max_cat_threshold=64, max_bin=args.bins,
            seed=args.seed, nthread=args.threads)
        objective_kwargs = {}
        if pairwise:
            from pairwise_auc import GlobalPairwiseObjective
            params.update(objective="reg:squarederror", base_score=0, boost_from_average=0)
            objective_kwargs["obj"] = GlobalPairwiseObjective(args.pairwise_negatives, args.seed)
        model = xgb.train(params, train_pool, num_boost_round=args.iterations,
            evals=[] if args.final or args.fixed_iterations else [(valid_pool,"valid")],
            early_stopping_rounds=None if args.final or args.fixed_iterations else args.patience, verbose_eval=100,
            **objective_kwargs)
        best = args.iterations if args.final or args.fixed_iterations else model.best_iteration+1
        prediction = model.predict(valid_pool, iteration_range=(0,best))
        if pairwise:
            from pairwise_auc import sigmoid
            prediction = sigmoid(prediction)
        model = model[:best]
        model.save_model(output / "model.ubj")
        gains = model.get_score(importance_type="gain")
        importance = [gains.get(c,0) for c in features]
    else:
        from catboost import CatBoostClassifier, Pool
        params = dict(iterations=args.iterations, learning_rate=args.learning_rate, depth=args.depth,
            l2_leaf_reg=args.l2, loss_function="Logloss", eval_metric="AUC", random_seed=args.seed,
            task_type="GPU", devices="0", gpu_ram_part=0.75, border_count=args.bins,
            thread_count=args.threads, allow_writing_files=False, verbose=100,
            max_ctr_complexity=args.ctr_complexity, one_hot_max_size=16, random_strength=args.random_strength)
        params.update(catboost_class_weight_params(positive_weight))
        train_pool = Pool(x_train, y_train, cat_features=cats, weight=weights)
        valid_pool = Pool(x_valid, None if args.final else y_valid, cat_features=cats)
        del x_train, x_valid
        gc.collect()
        model = CatBoostClassifier(**params)
        if args.final or args.fixed_iterations:
            model.fit(train_pool)
            best = args.iterations
        else:
            model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=args.patience, use_best_model=True)
            best = model.get_best_iteration() + 1
        # Completed boosting rounds include patience rounds later discarded by best-model selection.
        history = model.get_evals_result()
        logged_points = max((len(values) for dataset in history.values() for values in dataset.values()), default=0)
        # Fixed GPU fits can log metrics every five rounds; history length is then
        # not a round count. Without stopping, every requested tree was completed.
        completed = args.iterations if args.final or args.fixed_iterations else max(logged_points, int(model.tree_count_))
        iteration_metadata = {"completed_iterations": int(completed), "saved_tree_count": int(model.tree_count_),
                              "logged_metric_points": int(logged_points),
                              "selection_metric": "unweighted validation ROC AUC" if not (args.final or args.fixed_iterations) else None}
        if model.tree_count_ != best:
            raise AssertionError("Saved CatBoost trees do not match selected iteration")
        prediction = model.predict_proba(valid_pool)[:, 1]
        model.save_model(str(output / "model.cbm"))
        importance = model.feature_importances_
    if not pairwise:
        prediction = corrected_probability(prediction, args.negative_fraction, positive_weight)
    metrics = {} if args.final else {
        "auc": float(roc_auc_score(y_valid, prediction)),
        "average_precision": float(average_precision_score(y_valid, prediction)),
        **({} if pairwise else {"log_loss": float(log_loss(y_valid, prediction))})}
    specialist_metadata = {}
    if group_column is not None:
        group_labels = y_valid[evaluation_group_mask]
        group_prediction = prediction[evaluation_group_mask]
        group_metrics = {}
        if not args.final and len(np.unique(group_labels)) == 2:
            group_metrics = {
                "auc": float(roc_auc_score(group_labels, group_prediction)),
                "average_precision": float(average_precision_score(group_labels, group_prediction))}
        specialist_metadata = {
            "specialist_full_train_rows": specialist_full_train_count,
            "evaluation_scope": "all validation/test rows retained in original order for prediction alignment; specialist predictions interpretable only within training group",
            "training_group": {"column": group_column, "value": group_value,
                               "value_space": "encoded value in supplied parquet cache; not a decoded raw category; codes may differ across caches",
                               "target_statistics": "computed from original full training history before group selection"},
            "evaluation_group": {"rows": int(evaluation_group_mask.sum()), "metrics": group_metrics}}
    result = identifiers.with_columns(pl.Series("pctr", prediction))
    result.write_parquet(output / ("test_predictions.parquet" if args.final else "validation_predictions.parquet"))
    pd.DataFrame({"feature": features, "importance": importance}).sort_values(
        "importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    write_json(output / "experiment.json", {"args": vars(args), "params": params, "metrics": metrics,
        "best_iterations": int(best), "features": features, "categorical": cats,
        **iteration_metadata,
        "positive_weight": positive_weight, "effective_prior_correction_factor": effective_fraction,
        "train_rows": len(y_train), "full_train_rows": full_train_count, "evaluation_rows": len(y_valid),
        "validation_day": args.day, "elapsed_seconds": time.monotonic()-start,
        "target_encoding": ({"lodo": "training-label-only; training rows exclude their complete day",
                             "past": "training-label-only; every row uses strictly earlier dates for counts, clicks and global prior; no-history prior=0.01"}
                            .get(args.target_encoding)),
        "validation_row_sha256": hashlib.sha256(identifiers["row_index"].to_numpy().tobytes()).hexdigest(),
        "prediction_correction": "none; sigmoid of pairwise ranking margin" if pairwise else "uniform negative sampling and positive class-weight odds correction; effective factor=negative_fraction/positive_weight, applied once",
        "prediction_kind": "ranking_score" if pairwise else "probability",
        **specialist_metadata,
        **({"pairwise_objective": {"negatives_per_positive": args.pairwise_negatives,
            "seed": args.seed, "sampling": "uniform global training-fold negatives with replacement",
            "hessian": "diagonal approximation; floor=1e-6", "scale": "n_train/(2*num_pairs)"}} if pairwise else {}),
        "iteration_selection": "fixed before evaluation" if args.final or args.fixed_iterations else "early stopping on evaluation labels",
        "code_sha256": code_hash})
    print(json.dumps({"name": args.name, "best_iterations": int(best), "metrics": metrics,
                      "elapsed_seconds": time.monotonic()-start}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--features", type=Path, default=ROOT / "cache" / "features_v1.parquet")
    p.add_argument("--groups", default="base,sequence,counts,source")
    p.add_argument("--drop", default="")
    p.add_argument("--interest", type=Path)
    p.add_argument("--extra", type=Path, action="append", default=[])
    p.add_argument("--day", type=int, default=20191116)
    p.add_argument("--train-start-day", type=int, default=20191110)
    p.add_argument("--train-group-column", help="Train specialist on this cache column; requires --train-group-value")
    p.add_argument("--train-group-value", type=int, help="Encoded integer in supplied cache; validation rows remain unfiltered")
    p.add_argument("--half-life", type=float, default=0)
    p.add_argument("--model", choices=["lightgbm", "catboost", "xgboost"], default="lightgbm")
    p.add_argument("--xgb-objective", choices=["logloss", "pairwise"], default="logloss")
    p.add_argument("--pairwise-negatives", type=int, default=32)
    p.add_argument("--target-encoding", choices=["lodo", "past"])
    p.add_argument("--extended-te", action="store_true")
    p.add_argument("--te-rates-only", action="store_true",
                   help="Use only smoothed CTR target statistics as model features")
    p.add_argument("--negative-fraction", type=float, default=0.4)
    p.add_argument("--positive-weight", type=float, default=1.0,
                   help="CatBoost positive-class weight; predictions undo sampling/weighting with negative_fraction/positive_weight")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=0.04)
    p.add_argument("--leaves", type=int, default=31)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--min-child", type=int, default=150)
    p.add_argument("--l2", type=float, default=10)
    p.add_argument("--l1", type=float, default=0)
    p.add_argument("--colsample", type=float, default=0.85)
    p.add_argument("--cat-smooth", type=float, default=30)
    p.add_argument("--cat-mode", choices=["categorical", "numeric", "lowcard"], default="categorical")
    p.add_argument("--bins", type=int, default=127)
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--patience", type=int, default=150)
    p.add_argument("--random-strength", type=float, default=1)
    p.add_argument("--ctr-complexity", type=int, default=1)
    p.add_argument("--fixed-iterations", action="store_true")
    p.add_argument("--final", action="store_true")
    args = p.parse_args()
    if (args.train_group_column is None) != (args.train_group_value is None):
        p.error("train-group-column and train-group-value must be specified together")
    if args.pairwise_negatives < 1:
        p.error("pairwise-negatives must be positive")
    if args.xgb_objective == "pairwise" and (args.model != "xgboost" or args.half_life):
        p.error("pairwise requires --model xgboost and does not support --half-life/sample weights")
    if not 0 < args.negative_fraction <= 1:
        p.error("negative-fraction must be in (0, 1]")
    try:
        validate_positive_weight(args.model, args.negative_fraction, args.positive_weight)
    except ValueError as error:
        p.error(str(error))
    if args.final != (args.day == 20191117):
        p.error("--final is required exactly for the hidden test day 20191117")
    run(args)
