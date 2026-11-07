from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "humor_context.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
LABELS = ["non_humorous", "humorous"]

def make_model():
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=12000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=12000))
        ])),
        ("classifier", LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=42))
    ])

def evaluate(name, model, x_test, y_test):
    pred = model.predict(x_test)
    report = classification_report(y_test, pred, labels=LABELS, zero_division=0)
    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    print(f"\n--- {name.upper()} ---\n{report}\nAccuracy: {acc:.3f} | Macro F1: {f1:.3f}")
    cm = confusion_matrix(y_test, pred, labels=LABELS)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(values_format="d")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{name}.png", dpi=140)
    plt.close()
    return {"accuracy": acc, "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
            "f1_macro": f1, "classification_report": report}

def main():
    df = pd.read_csv(DATA_PATH).fillna("")
    for c in ("context", "text"):
        df[c] = df[c].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower().str.replace("-", "_").str.replace(" ", "_")
    df = df[df["label"].isin(LABELS) & (df["text"] != "")].drop_duplicates().copy()
    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both labels.")
    # Split by context so the paired responses to one scenario don't leak across train/test.
    groups = df["context"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    y_train, y_test = train_df["label"], test_df["label"]
    print(f"Rows: {len(df)} | train: {len(train_df)} | test: {len(test_df)}")
    print("Split is grouped by context; results may differ from earlier random-split metrics.")

    baseline = make_model()
    baseline.fit(train_df["text"].tolist(), y_train)
    joblib.dump(baseline, MODEL_DIR / "baseline.joblib")
    baseline_scores = evaluate("baseline", baseline, test_df["text"].tolist(), y_test)

    # Separate context and target with explicit field markers. Keep input identical at inference.
    train_x = [f"CONTEXT: {c} TARGET UTTERANCE: {t}" for c, t in zip(train_df["context"], train_df["text"])]
    test_x = [f"CONTEXT: {c} TARGET UTTERANCE: {t}" for c, t in zip(test_df["context"], test_df["text"])]
    context_model = make_model()
    context_model.fit(train_x, y_train)
    joblib.dump(context_model, MODEL_DIR / "context_aware.joblib")
    context_scores = evaluate("context_aware", context_model, test_x, y_test)

    metrics = {
        "dataset_rows_after_cleaning": len(df), "train_rows": len(train_df), "test_rows": len(test_df),
        "split_method": "GroupShuffleSplit by context to avoid paired-context leakage",
        "dataset_note": "Small illustrative dataset; metrics are preliminary and not generalizable.",
        "models": {"baseline": {"description": "Target utterance only", **baseline_scores},
                   "context_aware": {"description": "Context plus target utterance", **context_scores}}
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved models to {MODEL_DIR}; metrics to {OUTPUT_DIR / 'metrics.json'}")
    print("Do not claim the context-aware model is better unless the measured results support it.")

if __name__ == "__main__":
    main()
