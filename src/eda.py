"""Exploratory data analysis for the fake-news dataset.

The functions here deliberately use only pandas, NumPy, and matplotlib so the
project remains reproducible with the dependencies already used elsewhere.
"""

from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _top_terms(tokenized_docs, labels, label, n=12):
    counts = Counter(
        token
        for tokens, target in zip(tokenized_docs, labels)
        if int(target) == label
        for token in tokens
    )
    return counts.most_common(n)


def run_eda(df, tokenized_docs, labels, output_prefix):
    """Save a compact EDA figure, summary JSON, and class-term table.

    ``output_prefix`` follows the same convention as the pipeline outputs;
    e.g. ``outputs/run`` creates ``outputs/run_eda.png``.
    """
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=int)
    char_lengths = df["text"].astype(str).str.len().to_numpy()
    token_lengths = np.array([len(tokens) for tokens in tokenized_docs])
    counts = {"real": int((labels == 0).sum()), "fake": int((labels == 1).sum())}

    summary = {
        "documents": int(len(df)),
        "class_counts": counts,
        "missing_text_rows_removed": int(df["text"].isna().sum()),
        "article_length_characters": {
            "mean": round(float(char_lengths.mean()), 2),
            "median": round(float(np.median(char_lengths)), 2),
        },
        "article_length_tokens_after_preprocessing": {
            "mean": round(float(token_lengths.mean()), 2),
            "median": round(float(np.median(token_lengths)), 2),
        },
    }
    with open(f"{output_prefix}_eda_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    top_real = _top_terms(tokenized_docs, labels, 0)
    top_fake = _top_terms(tokenized_docs, labels, 1)
    with open(f"{output_prefix}_top_terms.csv", "w", encoding="utf-8") as handle:
        handle.write("class,term,count\n")
        for class_name, terms in (("real", top_real), ("fake", top_fake)):
            for term, count in terms:
                handle.write(f"{class_name},{term},{count}\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = ["#2a9d8f", "#e76f51"]
    axes[0, 0].bar(["Real", "Fake"], [counts["real"], counts["fake"]], color=colors)
    axes[0, 0].set_title("Class distribution")
    axes[0, 0].set_ylabel("Articles")

    for label, name, color in ((0, "Real", colors[0]), (1, "Fake", colors[1])):
        axes[0, 1].hist(char_lengths[labels == label], bins=20, alpha=0.65, label=name, color=color)
    axes[0, 1].set_title("Article length (characters)")
    axes[0, 1].set_xlabel("Characters")
    axes[0, 1].legend()

    for axis, title, terms, color in (
        (axes[1, 0], "Most frequent real-news terms", top_real, colors[0]),
        (axes[1, 1], "Most frequent fake-news terms", top_fake, colors[1]),
    ):
        words, values = zip(*terms) if terms else ([], [])
        axis.barh(list(reversed(words)), list(reversed(values)), color=color)
        axis.set_title(title)
        axis.set_xlabel("Occurrences")

    fig.suptitle("Exploratory Data Analysis", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_eda.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary
