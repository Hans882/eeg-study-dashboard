"""Leave-one-subject-out SVM training and evaluation (methodology 6.1, 8-10).

Produces:
  models/svm_workload.joblib   final model, trained on all but the demo subject
  models/demo_stream.npz       held-out subject's epochs, for the live replay
  models/loso_results.csv      per-subject scores for the report
  models/confusion_matrix.png  aggregated confusion matrix
"""
from __future__ import annotations

import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config as C

GRID = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]}


def make_svm(C_=1.0, gamma="scale") -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=C_, gamma=gamma,
                    class_weight="balanced", probability=True)),
    ])


def make_model(kind: str) -> Pipeline:
    if kind == "svm":
        return make_svm()
    if kind == "rf":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", RandomForestClassifier(
                             n_estimators=300, random_state=42,
                             class_weight="balanced", n_jobs=-1))])
    if kind == "dummy":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", DummyClassifier(strategy="most_frequent"))])
    raise ValueError(kind)


def loso(X, y, groups, kind="svm", tune=True):
    """Leave-one-subject-out CV. Returns per-fold rows and pooled predictions."""
    logo = LeaveOneGroupOut()
    rows, y_true_all, y_pred_all = [], [], []

    for i, (tr, te) in enumerate(logo.split(X, y, groups), 1):
        sid = int(groups[te][0])
        model = make_model(kind)

        if kind == "svm" and tune:
            # Inner CV is also grouped -- otherwise epochs from one subject
            # leak across the inner split and the chosen C/gamma are optimistic.
            inner = GroupKFold(n_splits=3)
            search = GridSearchCV(model, GRID, cv=inner, scoring="accuracy",
                                  n_jobs=-1, refit=True)
            search.fit(X[tr], y[tr], groups=groups[tr])
            model, best = search.best_estimator_, search.best_params_
        else:
            model.fit(X[tr], y[tr])
            best = {}

        pred = model.predict(X[te])
        p, r, f1, _ = precision_recall_fscore_support(
            y[te], pred, average="binary", zero_division=0)
        acc = accuracy_score(y[te], pred)
        rows.append({"subject": sid, "n_epochs": len(te), "accuracy": acc,
                     "precision": p, "recall": r, "f1": f1, **best})
        y_true_all.append(y[te])
        y_pred_all.append(pred)
        print(f"  [{i:2d}/{logo.get_n_splits(groups=groups)}] "
              f"sub{sid:02d}  acc={acc:.3f}  f1={f1:.3f}")

    return pd.DataFrame(rows), np.concatenate(y_true_all), np.concatenate(y_pred_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip the inner grid search (minutes instead of ~1h)")
    ap.add_argument("--holdout-subject", type=int, default=None,
                    help="subject reserved for the live demo replay "
                         "(default: highest subject id)")
    ap.add_argument("--baselines", action="store_true",
                    help="also run majority-class and Random Forest baselines")
    args = ap.parse_args()

    d = np.load(C.DATA_PROC / "features.npz", allow_pickle=True)
    X, y, groups, names = d["X"], d["y"], d["groups"], list(d["names"])
    print(f"Loaded {X.shape[0]} epochs x {X.shape[1]} features, "
          f"{len(np.unique(groups))} subjects\n")

    print("LOSO cross-validation (SVM, RBF):")
    res, y_true, y_pred = loso(X, y, groups, "svm", tune=not args.fast)
    res.to_csv(C.MODELS / "loso_results.csv", index=False)

    acc = res["accuracy"].mean()
    print(f"\nSVM  mean acc {acc:.3f} +/- {res['accuracy'].std():.3f}   "
          f"mean F1 {res['f1'].mean():.3f}")

    summary = {"svm_mean_accuracy": float(acc),
               "svm_std_accuracy": float(res["accuracy"].std()),
               "svm_mean_f1": float(res["f1"].mean())}

    if args.baselines:
        for kind, label in [("dummy", "Majority"), ("rf", "RandForest")]:
            r, _, _ = loso(X, y, groups, kind, tune=False)
            print(f"{label:11s} mean acc {r['accuracy'].mean():.3f}")
            summary[f"{kind}_mean_accuracy"] = float(r["accuracy"].mean())

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["low", "high"]).plot(
        cmap="Blues", colorbar=False)
    plt.title(f"Pooled LOSO confusion matrix (acc={accuracy_score(y_true, y_pred):.3f})")
    plt.tight_layout()
    plt.savefig(C.MODELS / "confusion_matrix.png", dpi=160)
    print(f"\nWrote {C.MODELS/'confusion_matrix.png'}")

    # --- final model, with one subject held out for an honest live demo ---
    demo = args.holdout_subject or int(groups.max())
    mask = groups != demo
    best = res.loc[res["accuracy"].idxmax()]
    final = make_svm(best.get("clf__C", 1.0), best.get("clf__gamma", "scale"))
    final.fit(X[mask], y[mask])
    joblib.dump({"model": final, "feature_names": names, "demo_subject": demo},
                C.MODELS / "svm_workload.joblib")

    idx = np.where(groups == demo)[0]  # already in recording order: low then high
    np.savez_compressed(C.MODELS / "demo_stream.npz", X=X[idx], y=y[idx])

    summary["demo_subject"] = demo
    (C.MODELS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved model (demo subject {demo} held out) and demo_stream.npz")


if __name__ == "__main__":
    main()
