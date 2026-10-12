"""
Apple Quality Classification - Training & Model Evaluation Script
Compares K-Nearest Neighbors (KNN) vs Gaussian Naive Bayes (GNB)
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def main():
    print("=" * 60)
    print(" [Apple Quality Prediction - Model Training Pipeline]")
    print("=" * 60)

    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), "applequality.csv")
    if not os.path.exists(data_path):
        data_path = "applequality.csv"

    print(f"\n[+] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"    Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop ID column if exists
    if "A_id" in df.columns:
        df = df.drop(columns=["A_id"])

    # Separate Features and Target
    X = df.drop(columns=["Quality"])
    y = df["Quality"]

    # Binary label mapping for ROC-AUC
    y_binary = y.map({"good": 1, "bad": 0})

    feature_names = list(X.columns)
    print(f"    Features ({len(feature_names)}): {', '.join(feature_names)}")
    print(f"    Target classes: {y.value_counts().to_dict()}")

    # 2. Train / Test Split (80% train, 20% test, stratified for balance)
    X_train, X_test, y_train, y_test, y_bin_train, y_bin_test = train_test_split(
        X, y, y_binary, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[+] Data split: Train set = {len(X_train)} samples, Test set = {len(X_test)} samples")

    # 3. Feature Scaling (StandardScaler) - crucial for distance-based algorithms like KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Models
    print("\n[+] Training Models...")
    # Model 1: K-Nearest Neighbors (k=5)
    knn_model = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn_model.fit(X_train_scaled, y_train)

    # Model 2: Gaussian Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)

    # 5. Model Evaluation
    y_pred_knn = knn_model.predict(X_test_scaled)
    y_proba_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

    y_pred_nb = nb_model.predict(X_test_scaled)
    y_proba_nb = nb_model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "KNN": {
            "Accuracy": accuracy_score(y_test, y_pred_knn),
            "Precision": precision_score(y_test, y_pred_knn, pos_label="good"),
            "Recall": recall_score(y_test, y_pred_knn, pos_label="good"),
            "F1-Score": f1_score(y_test, y_pred_knn, pos_label="good"),
            "ROC-AUC": roc_auc_score(y_bin_test, y_proba_knn),
        },
        "Naive Bayes": {
            "Accuracy": accuracy_score(y_test, y_pred_nb),
            "Precision": precision_score(y_test, y_pred_nb, pos_label="good"),
            "Recall": recall_score(y_test, y_pred_nb, pos_label="good"),
            "F1-Score": f1_score(y_test, y_pred_nb, pos_label="good"),
            "ROC-AUC": roc_auc_score(y_bin_test, y_proba_nb),
        },
    }

    print("\n" + "=" * 60)
    print(" Performance Benchmark Comparison")
    print("=" * 60)
    print(f"{'Metric':<15} | {'KNN (k=5)':<12} | {'Gaussian Naive Bayes':<20} | {'Advantage'}")
    print("-" * 60)
    for m in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        knn_val = metrics["KNN"][m]
        nb_val = metrics["Naive Bayes"][m]
        diff = (knn_val - nb_val) * 100
        winner = f"KNN (+{diff:.1f}%)" if diff > 0 else f"NB (+{-diff:.1f}%)"
        print(f"{m:<15} | {knn_val * 100:6.2f}%      | {nb_val * 100:6.2f}%               | {winner}")
    print("=" * 60)

    # 6. Save Artifacts & Models
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    static_plots_dir = os.path.join(os.path.dirname(__file__), "static", "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(static_plots_dir, exist_ok=True)

    joblib.dump(knn_model, os.path.join(models_dir, "knn_model.pkl"))
    joblib.dump(nb_model, os.path.join(models_dir, "nb_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    print(f"\n[+] Saved trained models & scaler to: {models_dir}")

    # 7. Generate Visualizations
    sns.set_theme(style="darkgrid")
    palette_dark = ["#10b981", "#3b82f6"]

    # Chart 1: Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    knn_scores = [metrics["KNN"][m] * 100 for m in metric_labels]
    nb_scores = [metrics["Naive Bayes"][m] * 100 for m in metric_labels]

    x = np.arange(len(metric_labels))
    width = 0.35

    rects1 = ax.bar(x - width / 2, knn_scores, width, label="K-Nearest Neighbors", color="#10b981", edgecolor="#059669")
    rects2 = ax.bar(x + width / 2, nb_scores, width, label="Gaussian Naive Bayes", color="#6366f1", edgecolor="#4f46e5")

    ax.set_ylabel("Score Percentage (%)", color="#f8fafc", fontsize=12)
    ax.set_title("Performance Comparison: KNN vs Naive Bayes", color="#f8fafc", fontsize=15, pad=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color="#f8fafc", fontsize=11)
    ax.tick_params(colors="#f8fafc")
    ax.set_ylim(60, 100)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=11)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", color="#f8fafc", fontsize=9, fontweight="bold")

    plt.tight_layout()
    comp_path = os.path.join(static_plots_dir, "model_comparison.png")
    plt.savefig(comp_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[+] Saved comparison plot: {comp_path}")

    # Chart 2: Confusion Matrices
    cm_knn = confusion_matrix(y_test, y_pred_knn, labels=["good", "bad"])
    cm_nb = confusion_matrix(y_test, y_pred_nb, labels=["good", "bad"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#0f172a")
    for ax, cm, title, color_map in zip(
        axes,
        [cm_knn, cm_nb],
        [f"KNN (Acc: {metrics['KNN']['Accuracy']*100:.1f}%)", f"Naive Bayes (Acc: {metrics['Naive Bayes']['Accuracy']*100:.1f}%)"],
        ["Greens", "Purples"]
    ):
        ax.set_facecolor("#1e293b")
        sns.heatmap(cm, annot=True, fmt="d", cmap=color_map, ax=ax, cbar=False,
                    xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"],
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(title, color="#f8fafc", fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("True Quality", color="#cbd5e1", fontsize=11)
        ax.set_xlabel("Predicted Quality", color="#cbd5e1", fontsize=11)
        ax.tick_params(colors="#f8fafc")

    plt.tight_layout()
    cm_path = os.path.join(static_plots_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[+] Saved confusion matrices plot: {cm_path}")

    # Chart 3: Feature Correlation Heatmap with Quality
    df_numeric = X.copy()
    df_numeric["Quality_Good"] = y_binary
    corr = df_numeric.corr()

    fig, ax = plt.subplots(figsize=(8.5, 6.5), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                linewidths=0.5, linecolor="#0f172a", annot_kws={"size": 9, "color": "#f8fafc"})
    ax.set_title("Feature Correlations & Relationship with Quality", color="#f8fafc", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors="#f8fafc")
    plt.tight_layout()
    corr_path = os.path.join(static_plots_dir, "feature_correlations.png")
    plt.savefig(corr_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[+] Saved feature correlations plot: {corr_path}")

    # Chart 4: Feature Distributions by Quality
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), facecolor="#0f172a")
    axes = axes.flatten()

    for idx, feature in enumerate(feature_names):
        ax = axes[idx]
        ax.set_facecolor("#1e293b")
        sns.kdeplot(data=df, x=feature, hue="Quality", palette={"good": "#10b981", "bad": "#f43f5e"},
                    fill=True, common_norm=False, alpha=0.4, ax=ax)
        ax.set_title(feature, color="#f8fafc", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        if ax.get_legend():
            ax.get_legend().remove()

    # Hide extra subplot (8th)
    axes[7].set_facecolor("#1e293b")
    axes[7].text(0.5, 0.5, "Apple Quality\nDistributions\n(Green = Good, Red = Bad)",
                 color="#f8fafc", ha="center", va="center", fontsize=11, fontweight="bold")
    axes[7].axis("off")

    plt.suptitle("Feature Density Distributions: Good vs Bad Apples", color="#f8fafc", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    dist_path = os.path.join(static_plots_dir, "feature_distributions.png")
    plt.savefig(dist_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[+] Saved feature distributions plot: {dist_path}")

    print("\n Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
