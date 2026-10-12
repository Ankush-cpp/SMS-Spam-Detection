"""
train_model.py
==============
Downloads the SMS Spam Collection dataset, preprocesses it, trains a
TF-IDF + Multinomial Naive Bayes pipeline, evaluates it on the held-out
test split, and saves the trained pipeline and evaluation artefacts to
the `model/` folder.

Run:
    python train_model.py
"""

import io
import zipfile
import pathlib
import json

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend — safe for any environment
import matplotlib.pyplot as plt
import seaborn as sns

from text_utils import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "model"
ASSETS_DIR = BASE_DIR / "assets"

for d in (DATA_DIR, MODEL_DIR, ASSETS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_DIR / "SMSSpamCollection"
MODEL_PATH   = MODEL_DIR / "spam_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# ── Dataset source (UCI ML Repository) ────────────────────────────────────
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
)


# ─────────────────────────────────────────────────────────────────────────
# 1. Download dataset
# ─────────────────────────────────────────────────────────────────────────
def download_dataset() -> pathlib.Path:
    """Download and extract the SMS Spam Collection if not already present."""
    if DATASET_PATH.exists():
        print(f"[INFO] Dataset already present at {DATASET_PATH}")
        return DATASET_PATH

    print(f"[INFO] Downloading dataset from {DATASET_URL} …")
    try:
        resp = requests.get(DATASET_URL, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            f"\n[ERROR] Could not download dataset: {exc}\n"
            "Please download it manually from:\n"
            "  https://archive.ics.uci.edu/dataset/228/sms+spam+collection\n"
            f"Extract 'SMSSpamCollection' into  {DATA_DIR}/\n"
        ) from exc

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # The archive contains 'SMSSpamCollection' (tab-separated, no header)
        names = zf.namelist()
        print(f"[INFO] Archive contents: {names}")
        target = next((n for n in names if "SMSSpamCollection" in n and "." not in n.split("/")[-1]), None)
        if target is None:
            # fallback: extract all and look for the file
            zf.extractall(DATA_DIR)
        else:
            with zf.open(target) as src, open(DATASET_PATH, "wb") as dst:
                dst.write(src.read())

    if not DATASET_PATH.exists():
        # Try to locate the extracted file
        candidates = list(DATA_DIR.rglob("SMSSpamCollection"))
        if candidates:
            candidates[0].rename(DATASET_PATH)
        else:
            raise SystemExit(
                f"[ERROR] 'SMSSpamCollection' not found after extraction in {DATA_DIR}.\n"
                "Please place it there manually."
            )

    print(f"[INFO] Dataset saved to {DATASET_PATH}")
    return DATASET_PATH


# ─────────────────────────────────────────────────────────────────────────
# 2. Load & preprocess
# ─────────────────────────────────────────────────────────────────────────
def load_data(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",
    )
    print(f"[INFO] Loaded {len(df)} rows  |  spam={df['label'].eq('spam').sum()}  ham={df['label'].eq('ham').sum()}")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


# ─────────────────────────────────────────────────────────────────────────
# 3. Train
# ─────────────────────────────────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),
            max_features=10_000,
            sublinear_tf=True,
        )),
        ("clf", MultinomialNB(alpha=0.1)),
    ])


# ─────────────────────────────────────────────────────────────────────────
# 4. Evaluate & save artefacts
# ─────────────────────────────────────────────────────────────────────────
def evaluate_and_save(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
    }
    print("\n[RESULTS]")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")

    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"])
    print("\nClassification Report:\n", report)

    # ── Confusion matrix plot ────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path = ASSETS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved to {cm_path}")

    # ── Metrics bar chart ────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    metric_names  = list(metrics.keys())
    metric_values = list(metrics.values())
    bars = ax2.bar(metric_names, metric_values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_title("Model Performance Metrics", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, metric_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.4f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    metrics_path = ASSETS_DIR / "metrics_bar.png"
    fig2.savefig(metrics_path, dpi=120)
    plt.close(fig2)
    print(f"[INFO] Metrics bar chart saved to {metrics_path}")

    # ── Save metrics JSON ────────────────────────────────────────────────
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ─────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────
def main():
    path = download_dataset()
    df   = load_data(path)

    X = df["message"]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("[INFO] Model trained.")

    evaluate_and_save(pipeline, X_test, y_test)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"[INFO] Pipeline saved to {MODEL_PATH}")
    print("\n[DONE] Training complete. Run:  streamlit run app.py")


if __name__ == "__main__":
    main()
