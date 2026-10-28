"""Compress source-domain interest sequences; no advertisement labels used."""
import json
import time
import joblib
import numpy as np
import polars as pl
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from common import ROOT, write_json


if __name__ == "__main__":
    start = time.monotonic()
    source_cols = ["u_userId", "e_et", "i_cat", "label", "u_newsCatInterests", "u_newsCatInterestsST", "u_click_ca2_news"]
    src = pl.concat([pl.scan_csv(ROOT / "2026ml" / filename, infer_schema_length=10000).select(source_cols)
                     for filename in ["train_feeds.csv", "test_feeds.csv"]]).collect(engine="streaming")
    parts = [pl.concat_str([pl.lit("v"), pl.col("i_cat").cast(pl.String)]),
             pl.when(pl.col("label") == 1).then(pl.concat_str([pl.lit("c"), pl.col("i_cat").cast(pl.String)])).otherwise(pl.lit(""))]
    for col, prefix in [("u_newsCatInterests", "l"), ("u_newsCatInterestsST", "s"), ("u_click_ca2_news", "r")]:
        parts.append(pl.concat_str([pl.lit(prefix), pl.col(col).str.replace_all(r"\^", " " + prefix)]))
    src = src.with_columns(pl.concat_str(parts, separator=" ").alias("document"))
    daily = src.group_by(["u_userId", "e_et"]).agg(pl.col("document").str.join(" ")).sort(["u_userId", "e_et"])
    del src
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=3, sublinear_tf=True, dtype=np.float32)
    fit_mask = daily["e_et"].to_numpy() <= 20191114
    documents = daily["document"].to_numpy()
    vectorizer.fit(documents[fit_mask])
    matrix = vectorizer.transform(documents)
    svd = TruncatedSVD(n_components=24, n_iter=7, random_state=2026)
    svd.fit(matrix[fit_mask])
    embedding = svd.transform(matrix).astype("float32")
    features = [f"interest_svd_{i:02d}" for i in range(24)]
    vectors = daily.select(pl.col("u_userId").alias("user_id"), pl.col("e_et").alias("pt_d")).hstack(pl.DataFrame(embedding, schema=features))
    vectors = vectors.with_columns(pl.Series("interest_distinct_tokens", np.diff(matrix.indptr).astype("float32")))
    features.append("interest_distinct_tokens")
    ads = pl.concat([pl.scan_csv(ROOT / "2026ml" / filename, infer_schema_length=10000).select(["user_id", "pt_d"])
                      .with_row_index("row_index").with_columns(pl.lit(is_test).cast(pl.Int8).alias("is_test"))
                     for filename, is_test in [("train.csv", 0), ("final_test.csv", 1)]])
    output = ROOT / "cache" / "interest_rows.parquet"
    ads.join(vectors.lazy(), on=["user_id", "pt_d"], how="left", maintain_order="left").select(
        ["row_index", "is_test"] + features).with_columns([pl.col(c).fill_null(0) for c in features]).sink_parquet(output, compression="zstd")
    joblib.dump({"vectorizer": vectorizer, "svd": svd}, ROOT / "cache" / "interest_transform.joblib")
    write_json(output.with_suffix(".json"), {"features": features, "fit_source_dates_max": 20191114,
        "target_domain_labels_used": False, "vocabulary": len(vectorizer.vocabulary_),
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "elapsed_seconds": time.monotonic()-start})
    print(f"Interest features ready: {output}; {time.monotonic()-start:.1f}s", flush=True)
