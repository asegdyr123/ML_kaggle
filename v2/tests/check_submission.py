"""Original-row alignment must survive duplicate business IDs and shuffled storage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl
from export_submission import blend_predictions

expected = pl.DataFrame({"row_index": [0, 1, 2], "log_id": [5, 5, 8], "pt_d": [20191117]*3})
a = expected.with_columns(pl.Series("pctr", [.1, .2, .3]), pl.lit(-1).alias("label"))
b = expected.with_columns(pl.Series("pctr", [.3, .4, .5]), pl.lit(-1).alias("label")).reverse()
assert np.allclose(blend_predictions([a, b], [.5, .5], expected), [.2, .3, .4])
bad_frames = [a.head(2), a.with_columns(pl.Series("row_index", [0, 0, 2])),
              a.with_columns(pl.Series("pctr", [.1, float("nan"), .3])),
              a.with_columns(pl.lit(0).alias("label")),
              a.with_columns(pl.lit(20191116).alias("pt_d"))]
for bad in bad_frames:
    try:
        blend_predictions([bad], [1.], expected)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid predictions accepted")
print("PASS: duplicate log_id preserved; storage order restored; incomplete/invalid predictions rejected")
