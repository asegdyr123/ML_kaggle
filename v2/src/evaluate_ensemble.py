"""Evaluate a specified blend on aligned held-out rows; optional user-cluster bootstrap."""
import argparse
import json
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from common import ROOT, write_json


def read_prediction(name):
    frame = pl.read_parquet(ROOT / "outputs" / name / "validation_predictions.parquet").sort("row_index")
    if frame["row_index"].n_unique() != len(frame):
        raise ValueError(f"Duplicate row_index in {name}")
    if frame["label"].null_count() or not set(frame["label"].unique().to_list()) <= {0, 1}:
        raise ValueError(f"Invalid validation labels in {name}")
    values = frame["pctr"].to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"Invalid probabilities in {name}")
    return frame


def metric(y, p):
    return {"auc":float(roc_auc_score(y,p)), "average_precision":float(average_precision_score(y,p)),
            "log_loss":float(log_loss(y,p))}


def ranked_auc_setup(y, p, users):
    order = np.argsort(p, kind="stable")
    starts = np.r_[0,np.flatnonzero(np.diff(p[order]))+1]
    return y[order].astype("float64"), users[order], starts


def weighted_auc(setup, cluster_weights):
    y, users, starts = setup
    weights = cluster_weights[users]
    positive = np.add.reduceat(weights*y, starts)
    negative = np.add.reduceat(weights*(1-y), starts)
    denominator = positive.sum()*negative.sum()
    return float(np.dot(positive, np.cumsum(negative)-0.5*negative)/denominator)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--name",required=True)
    parser.add_argument("--members",nargs="+",required=True)
    parser.add_argument("--weights",type=float,nargs="+")
    parser.add_argument("--reference")
    parser.add_argument("--bootstrap",type=int,default=0)
    args=parser.parse_args()
    weights=np.array(args.weights or [1/len(args.members)]*len(args.members))
    if len(weights)!=len(args.members) or (weights<0).any() or not np.isclose(weights.sum(),1):
        parser.error("Provide one nonnegative weight per member, summing to one")
    frames=[read_prediction(n) for n in args.members]
    anchor=frames[0]
    for frame in frames[1:]:
        for col in ["row_index","log_id","label","pt_d"]:
            assert anchor[col].equals(frame[col]), f"Prediction alignment mismatch: {col}"
    prediction=sum(w*frame["pctr"].to_numpy() for w,frame in zip(weights,frames))
    y=anchor["label"].to_numpy()
    result={"name":args.name,"members":args.members,"weights":weights.tolist(),"metrics":metric(y,prediction),
            "rows":len(anchor),"positive_rows":int(y.sum())}
    if args.reference:
        reference=read_prediction(args.reference)
        for col in ["row_index", "log_id", "label", "pt_d"]:
            assert reference[col].equals(anchor[col]), f"Reference alignment mismatch: {col}"
        refp=reference["pctr"].to_numpy()
        result["reference"]={"name":args.reference,"metrics":metric(y,refp)}
        result["auc_gain"]=result["metrics"]["auc"]-result["reference"]["metrics"]["auc"]
    if args.bootstrap:
        identifiers=pl.scan_parquet(ROOT/"cache"/"features_v1.parquet").filter(pl.col("is_test")==0).select(["row_index","user_id"]).collect()
        aligned=anchor.select("row_index").join(identifiers,on="row_index",how="left",maintain_order="left",validate="1:1")
        assert aligned["user_id"].null_count() == 0, "Missing bootstrap cluster IDs"
        _,users=np.unique(aligned["user_id"].to_numpy(),return_inverse=True)
        setup=ranked_auc_setup(y,prediction,users)
        assert np.isclose(weighted_auc(setup,np.ones(users.max()+1)),result["metrics"]["auc"])
        refsetup=ranked_auc_setup(y,refp,users) if args.reference else None
        rng=np.random.default_rng(2026)
        scores=[];differences=[]
        for _ in range(args.bootstrap):
            w=rng.poisson(1,size=users.max()+1)
            score=weighted_auc(setup,w);scores.append(score)
            if refsetup is not None:
                differences.append(score-weighted_auc(refsetup,w))
        result["bootstrap"]={"method":"user-cluster Poisson bootstrap", "replicates":args.bootstrap,
            "auc_95pct_interval":np.quantile(scores,[0.025,0.975]).tolist(),
            "note":"Describes this local holdout; does not adjust for model-selection bias"}
        if differences:
            result["bootstrap"]["auc_gain_95pct_interval"]=np.quantile(differences,[0.025,0.975]).tolist()
    out=ROOT/"outputs"/args.name
    out.mkdir(exist_ok=True,parents=True)
    anchor.select([c for c in ["row_index","log_id","label","pt_d"] if c in anchor.columns]).with_columns(pl.Series("pctr",prediction)).write_parquet(out/"validation_predictions.parquet")
    write_json(out/"ensemble.json",result)
    print(json.dumps(result,indent=2))
