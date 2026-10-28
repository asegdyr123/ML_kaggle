"""Check label exclusion and tied-score cluster AUC against independent expectations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from evaluate_ensemble import ranked_auc_setup, weighted_auc
from run_experiment import add_target_stats

rng = np.random.default_rng(31)
y = rng.integers(0, 2, 200)
p = np.round(rng.random(200), 1)
users = rng.integers(0, 20, 200)
w = rng.integers(1, 5, 20)
assert np.isclose(weighted_auc(ranked_auc_setup(y, p, users), w),
                  roc_auc_score(y, p, sample_weight=w[users]))
f = pl.DataFrame({"user_id": [1,1,2,1,2,3,1,4],
                  "pt_d": [11,11,11,12,12,12,13,13],
                  "label": [1,0,0,0,1,0,1,0]})
fit = f.filter(pl.col("pt_d") < 13)
a, names = add_target_stats(f, fit, 13, "lodo", [["user_id"]])
changed = f.with_columns(pl.when(pl.col("pt_d") == 13).then(1-pl.col("label"))
                         .otherwise(pl.col("label")).alias("label"))
b, _ = add_target_stats(changed, fit, 13, "lodo", [["user_id"]])
assert a.select(names).equals(b.select(names))
assert a["te_user_id_clicks"].to_list() == [0.,0.,1.,1.,0.,0.,1.,0.]
assert a["te_user_id_n"].to_list() == [1.,1.,1.,2.,1.,0.,3.,0.]
try:
    add_target_stats(f, f, 13, "lodo", [["user_id"]])
except AssertionError:
    pass
else:
    raise AssertionError("Cutoff guard failed")
print("PASS: tied weighted AUC; held-out-label invariance; complete-day exclusion; cutoff guard")
