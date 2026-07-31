"""
FakeCheck AI — multi-page web interface for the from-scratch fake news
detection pipeline.

Run:
    python3 app.py
Then open http://127.0.0.1:5000
"""

import io
import csv
import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, session, Response

from src.preprocessing import preprocess
from src.features import TfidfVectorizerScratch
from src.models import KNNScratch, LogisticRegressionScratch, RandomForestScratch, SimpleNeuralNetScratch
from src.evaluate import classification_metrics

app = Flask(__name__)
app.secret_key = "dev-key-change-in-production"

MODEL_CHOICES = {
    "logreg": "Logistic Regression",
    "knn": "KNN",
    "rf": "Random Forest",
    "nn": "Neural Network",
}

EXAMPLES = [
    "The city council in Austin voted to fund additional resources for public transit this year.",
    "You won't BELIEVE what NASA is hiding about the moon landing -- shocking truth revealed!!!",
    "Researchers at a state university published a peer-reviewed study on renewable energy adoption.",
    "BREAKING: secret documents PROVE a conspiracy, share before it's deleted!!!",
]

PIPELINES = {}  # feature_type -> {vectorizer, models, metrics, best_model}


def manual_train_test_split(X, y, test_size=0.2, seed=42):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def build_pipeline(tokenized, y, use_bigrams: bool):
    vectorizer = TfidfVectorizerScratch(max_features=3000, min_df=2, use_bigrams=use_bigrams).fit(tokenized)
    X = vectorizer.transform(tokenized)
    X_train, X_test, y_train, y_test = manual_train_test_split(X, y)

    models = {
        "knn": KNNScratch(k=5),
        "logreg": LogisticRegressionScratch(lr=0.5, n_iters=400),
        "rf": RandomForestScratch(n_trees=15, max_depth=8),
        "nn": SimpleNeuralNetScratch(hidden_size=32, lr=0.05, n_iters=300),
    }
    metrics = {}
    for key, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics[MODEL_CHOICES[key]] = classification_metrics(y_test, preds)

    best_model = max(metrics.keys(), key=lambda k: metrics[k]["f1"])

    return {
        "vectorizer": vectorizer,
        "models": models,
        "metrics": metrics,
        "best_model": best_model,
        "n_docs": len(tokenized),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
    }


def load_data_and_train():
    print("Loading data/news.csv and training all pipelines (one-time)...")
    df = pd.read_csv("data/news.csv").dropna(subset=["text", "label"]).reset_index(drop=True)
    tokenized = [preprocess(t) for t in df["text"]]
    y = df["label"].astype(int).to_numpy()

    PIPELINES["tfidf"] = build_pipeline(tokenized, y, use_bigrams=False)
    PIPELINES["tfidf_bigram"] = build_pipeline(tokenized, y, use_bigrams=True)
    print("All pipelines ready.")


def get_top_words(pipeline, tokens, top_n=8):
    """Only meaningful for logistic regression: contribution = tfidf_value * weight."""
    vec = pipeline["vectorizer"]
    lr = pipeline["models"]["logreg"]
    x = vec.transform([tokens])[0]
    nonzero = np.nonzero(x)[0]
    contribs = [(vec.vocab.idx2word[j], float(x[j] * lr.w[j])) for j in nonzero]
    contribs.sort(key=lambda t: -abs(t[1]))
    return contribs[:top_n]


@app.route("/", methods=["GET"])
def home():
    p = PIPELINES["tfidf"]
    stats = {
        "n_docs": p["n_docs"],
        "vocab_size": len(p["vectorizer"].vocab),
        "n_real": p["n_real"],
        "n_fake": p["n_fake"],
        "best_model": p["best_model"],
    }
    return render_template(
        "index.html", active="home", model_choices=MODEL_CHOICES,
        selected_model="logreg", feature_type="tfidf", examples=EXAMPLES, stats=stats,
    )


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "")
    model_key = request.form.get("model", "logreg")
    feature_type = request.form.get("feature_type", "tfidf")
    pipeline = PIPELINES["tfidf_bigram"] if feature_type == "tfidf_bigram" else PIPELINES["tfidf"]

    tokens = preprocess(text)
    X = pipeline["vectorizer"].transform([tokens])
    model = pipeline["models"][model_key]

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0])
    else:
        pred = int(model.predict(X)[0])
        prob = 0.9 if pred == 1 else 0.1  # KNN/RF: approximate confidence display

    result = "FAKE" if prob >= 0.5 else "REAL"
    confidence = prob if prob >= 0.5 else 1 - prob
    top_words = get_top_words(pipeline, tokens) if model_key == "logreg" and tokens else None

    hist = session.get("history", [])
    hist.insert(0, {
        "text": text, "prediction": result, "prob": prob,
        "model": MODEL_CHOICES[model_key],
        "time": datetime.datetime.now().strftime("%b %d, %H:%M"),
    })
    session["history"] = hist[:20]

    p = PIPELINES["tfidf"]
    stats = {
        "n_docs": p["n_docs"], "vocab_size": len(p["vectorizer"].vocab),
        "n_real": p["n_real"], "n_fake": p["n_fake"], "best_model": p["best_model"],
    }
    return render_template(
        "index.html", active="home", model_choices=MODEL_CHOICES,
        selected_model=model_key, feature_type=feature_type, examples=EXAMPLES, stats=stats,
        text=text, result=result, prob=prob, confidence=confidence, top_words=top_words,
    )


@app.route("/dashboard")
def dashboard():
    p = PIPELINES["tfidf"]
    metrics = p["metrics"]
    metric_names = ["accuracy", "precision", "recall", "f1"]
    model_names = list(metrics.keys())
    data_by_metric = {m: [metrics[name][m] for name in model_names] for m in metric_names}
    best_model = p["best_model"]
    cm = metrics[best_model]["confusion_matrix"].tolist()

    return render_template(
        "dashboard.html", active="dashboard", metrics=metrics, best_model=best_model,
        metric_names=metric_names, model_names=model_names, data_by_metric=data_by_metric,
        confusion_matrix=cm,
    )


@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html", active="batch", model_choices=MODEL_CHOICES,
                                selected_model="logreg", rows=None, error=None)

    model_key = request.form.get("model", "logreg")
    file = request.files.get("file")
    if not file or file.filename == "":
        return render_template("batch.html", active="batch", model_choices=MODEL_CHOICES,
                                selected_model=model_key, rows=None, error="Please choose a CSV file.")

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return render_template("batch.html", active="batch", model_choices=MODEL_CHOICES,
                                selected_model=model_key, rows=None, error=f"Could not read CSV: {e}")

    if "text" not in df.columns:
        return render_template("batch.html", active="batch", model_choices=MODEL_CHOICES,
                                selected_model=model_key, rows=None,
                                error="CSV must contain a 'text' column.")

    df = df.dropna(subset=["text"]).reset_index(drop=True)
    pipeline = PIPELINES["tfidf"]
    model = pipeline["models"][model_key]

    rows = []
    for text in df["text"].astype(str):
        tokens = preprocess(text)
        X = pipeline["vectorizer"].transform([tokens])
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0])
        else:
            pred = int(model.predict(X)[0])
            prob = 0.9 if pred == 1 else 0.1
        rows.append({"text": text, "prediction": "FAKE" if prob >= 0.5 else "REAL", "prob": prob})

    session["last_batch"] = rows
    return render_template("batch.html", active="batch", model_choices=MODEL_CHOICES,
                            selected_model=model_key, rows=rows, error=None)


@app.route("/batch/download", methods=["POST", "GET"])
def batch_download():
    rows = session.get("last_batch", [])
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["text", "prediction", "prob_fake"])
    for r in rows:
        writer.writerow([r["text"], r["prediction"], f"{r['prob']:.4f}"])
    return Response(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_results.csv"},
    )


@app.route("/history")
def history():
    return render_template("history.html", active="history", items=session.get("history", []))


@app.route("/history/clear", methods=["POST"])
def clear_history():
    session["history"] = []
    return redirect(url_for("history"))


@app.route("/about")
def about():
    return render_template("about.html", active="about")


if __name__ == "__main__":
    load_data_and_train()
    app.run(debug=False, host="127.0.0.1", port=5000)
