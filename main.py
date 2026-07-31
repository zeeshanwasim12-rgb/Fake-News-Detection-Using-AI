"""
AI-Powered Fake News Detection Using Text Classification
==========================================================
Full from-scratch pipeline: preprocessing -> feature extraction ->
model training -> evaluation. Run with:

    python main.py --data data/news.csv --features tfidf

See README.md for full instructions.
"""

import argparse
import time
import numpy as np
import pandas as pd

from src.preprocessing import preprocess
from src.features import Vocabulary, BagOfWords, TfidfVectorizerScratch, CooccurrenceEmbedding
from src.models import (
    KNNScratch,
    LogisticRegressionScratch,
    RandomForestScratch,
    SimpleNeuralNetScratch,
)
from src.evaluate import classification_metrics, print_report, plot_results
from src.evaluate import plot_confusion_matrices
from src.eda import run_eda


def manual_train_test_split(X, y, test_size=0.2, seed=42):
    """Random split implemented with numpy only (no sklearn.model_selection)."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main(data_path: str, feature_type: str, max_features: int, output_prefix: str):
    t0 = time.time()

    # ---------------- Week 1: Load & Clean ----------------
    print("Week 1: Loading & cleaning data...")
    df = pd.read_csv(data_path)
    assert "text" in df.columns and "label" in df.columns, \
        "CSV must contain 'text' and 'label' columns"
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

    tokenized = [preprocess(t) for t in df["text"]]
    y = df["label"].astype(int).to_numpy()
    print(f"  {len(tokenized)} documents loaded. Example tokens: {tokenized[0][:10]}")

    print("  Saving exploratory data analysis outputs...")
    eda_summary = run_eda(df, tokenized, y, output_prefix)
    print(f"  EDA: {eda_summary['documents']} documents; class counts {eda_summary['class_counts']}")

    # ---------------- Week 2: Feature Engineering ----------------
    print(f"\nWeek 2: Building '{feature_type}' features (max_features={max_features})...")
    vocab = Vocabulary(max_features=max_features, min_df=2).fit(tokenized)
    print(f"  Vocabulary size: {len(vocab)}")

    if feature_type == "bow":
        X = BagOfWords(vocab).transform(tokenized)
    elif feature_type == "tfidf":
        X = TfidfVectorizerScratch(max_features=max_features, min_df=2).fit_transform(tokenized)
    elif feature_type == "embedding":
        emb = CooccurrenceEmbedding(vocab, window=4, dim=50).fit(tokenized)
        X = emb.transform(tokenized)
    else:
        raise ValueError("feature_type must be one of: bow, tfidf, embedding")

    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=0.2)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ---------------- Week 3: Model Building ----------------
    print("\nWeek 3: Training models (KNN, Logistic Regression, Random Forest, Neural Net)...")
    models = {
        "KNN": KNNScratch(k=5),
        "LogisticRegression": LogisticRegressionScratch(lr=0.5, n_iters=300),
        "RandomForest": RandomForestScratch(n_trees=15, max_depth=8),
        "NeuralNet": SimpleNeuralNetScratch(hidden_size=32, lr=0.05, n_iters=300),
    }

    results = {}
    for name, model in models.items():
        t1 = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = classification_metrics(y_test, preds)
        results[name] = metrics
        print_report(name, metrics)
        print(f"  (trained in {time.time() - t1:.2f}s)")

    # ---------------- Week 4: Evaluate & Visualize ----------------
    print("\nWeek 4: Saving comparison chart...")
    chart_path = f"{output_prefix}_results.png"
    plot_results(results, chart_path)
    print(f"  Saved chart to {chart_path}")
    confusion_path = f"{output_prefix}_confusion_matrices.png"
    plot_confusion_matrices(results, confusion_path)
    print(f"  Saved confusion matrices to {confusion_path}")

    summary = pd.DataFrame({
        name: {k: v for k, v in m.items() if k != "confusion_matrix"}
        for name, m in results.items()
    }).T
    summary_path = f"{output_prefix}_summary.csv"
    summary.to_csv(summary_path)
    print(f"  Saved metrics summary to {summary_path}")
    print(summary)

    print(f"\nTotal pipeline time: {time.time() - t0:.2f}s")
    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake News Detection - from-scratch pipeline")
    parser.add_argument("--data", default="data/news.csv", help="Path to CSV with text,label columns")
    parser.add_argument("--features", default="tfidf", choices=["bow", "tfidf", "embedding"],
                         help="Feature extraction method")
    parser.add_argument("--max_features", type=int, default=3000)
    parser.add_argument("--output_prefix", default="outputs/run")
    args = parser.parse_args()

    main(args.data, args.features, args.max_features, args.output_prefix)
