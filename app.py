"""
Apple Quality Prediction Web Application
Flask Backend Server
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CSV_PATH = os.path.join(BASE_DIR, "applequality.csv")

# Load models and scaler
try:
    knn_model = joblib.load(os.path.join(MODELS_DIR, "knn_model.pkl"))
    nb_model = joblib.load(os.path.join(MODELS_DIR, "nb_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    print("[+] Pre-trained models loaded successfully.")
except Exception as e:
    print(f"[!] Warning: Could not load saved models ({e}). Retraining...")
    from train_and_evaluate import main as retrain_models
    retrain_models()
    knn_model = joblib.load(os.path.join(MODELS_DIR, "knn_model.pkl"))
    nb_model = joblib.load(os.path.join(MODELS_DIR, "nb_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

# Load dataset for sampling and statistics
dataset_df = pd.read_csv(CSV_PATH)
if "A_id" in dataset_df.columns:
    features_df = dataset_df.drop(columns=["A_id", "Quality"])
else:
    features_df = dataset_df.drop(columns=["Quality"])

FEATURE_NAMES = list(features_df.columns)
FEATURE_STATS = {
    col: {
        "min": round(float(features_df[col].min()), 2),
        "max": round(float(features_df[col].max()), 2),
        "mean": round(float(features_df[col].mean()), 2),
        "std": round(float(features_df[col].std()), 2),
    }
    for col in FEATURE_NAMES
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        model_choice = data.get("model", "both").lower()

        # Extract features in strict order
        input_values = []
        for feat in FEATURE_NAMES:
            if feat not in data:
                return jsonify({"error": f"Missing feature: {feat}"}), 400
            input_values.append(float(data[feat]))

        # Transform features
        input_array = np.array([input_values])
        input_scaled = scaler.transform(input_array)

        # KNN Prediction
        knn_pred = knn_model.predict(input_scaled)[0]
        knn_proba = knn_model.predict_proba(input_scaled)[0]
        knn_classes = list(knn_model.classes_)
        knn_good_idx = knn_classes.index("good") if "good" in knn_classes else 0
        knn_good_prob = float(knn_proba[knn_good_idx])
        knn_bad_prob = 1.0 - knn_good_prob

        # Naive Bayes Prediction
        nb_pred = nb_model.predict(input_scaled)[0]
        nb_proba = nb_model.predict_proba(input_scaled)[0]
        nb_classes = list(nb_model.classes_)
        nb_good_idx = nb_classes.index("good") if "good" in nb_classes else 0
        nb_good_prob = float(nb_proba[nb_good_idx])
        nb_bad_prob = 1.0 - nb_good_prob

        # Primary decision based on user's chosen model
        if model_choice == "naive_bayes":
            primary_pred = nb_pred
            primary_conf = round(max(nb_good_prob, nb_bad_prob) * 100, 1)
        else:
            primary_pred = knn_pred
            primary_conf = round(max(knn_good_prob, knn_bad_prob) * 100, 1)

        response = {
            "status": "success",
            "primary_prediction": primary_pred,
            "primary_confidence": primary_conf,
            "selected_model": model_choice,
            "knn": {
                "prediction": knn_pred,
                "confidence": round(max(knn_good_prob, knn_bad_prob) * 100, 1),
                "prob_good": round(knn_good_prob * 100, 1),
                "prob_bad": round(knn_bad_prob * 100, 1),
            },
            "naive_bayes": {
                "prediction": nb_pred,
                "confidence": round(max(nb_good_prob, nb_bad_prob) * 100, 1),
                "prob_good": round(nb_good_prob * 100, 1),
                "prob_bad": round(nb_bad_prob * 100, 1),
            },
            "features": dict(zip(FEATURE_NAMES, input_values)),
        }
        return jsonify(response)

    except Exception as err:
        return jsonify({"error": str(err)}), 500


@app.route("/api/sample", methods=["GET"])
def get_sample():
    quality_filter = request.args.get("quality", "").lower()
    if quality_filter in ["good", "bad"]:
        sub_df = dataset_df[dataset_df["Quality"] == quality_filter]
    else:
        sub_df = dataset_df

    sample_row = sub_df.sample(n=1).iloc[0].to_dict()
    # Remove internal IDs
    if "A_id" in sample_row:
        sample_row["id"] = int(sample_row.pop("A_id"))
    
    # Round float features
    for col in FEATURE_NAMES:
        sample_row[col] = round(float(sample_row[col]), 2)

    return jsonify(sample_row)


@app.route("/api/stats", methods=["GET"])
def get_stats():
    total_samples = len(dataset_df)
    counts = dataset_df["Quality"].value_counts().to_dict()
    return jsonify({
        "total_samples": total_samples,
        "class_distribution": counts,
        "feature_stats": FEATURE_STATS,
        "metrics": {
            "knn": {"accuracy": 87.50, "precision": 86.80, "recall": 88.53, "f1": 87.65, "auc": 94.36},
            "naive_bayes": {"accuracy": 74.00, "precision": 73.59, "recall": 75.06, "f1": 74.32, "auc": 83.05}
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Apple Quality App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
