# 🍏 Apple Quality Prediction • AI Classification System

An end-to-end Machine Learning classification and interactive intelligence platform designed to predict whether an apple is of **Good** or **Bad** quality based on its sensory and physical attributes.

This project implements, evaluates, and compares two core machine learning algorithms:
1. **K-Nearest Neighbors (KNN)** with Feature Normalization (`StandardScaler`)
2. **Gaussian Naive Bayes (GNB)** Probabilistic Classifier

It features an interactive **Web Application Dashboard**, clean standalone Python scripts, publication-grade analytical plots, and a Jupyter Notebook.

---

## 📌 Project Overview & Highlights

* **Dataset Size:** 4,000 apple records (perfectly balanced: 2,004 Good vs 1,996 Bad).
* **Target Variable:** `Quality` (`good` or `bad`).
* **Input Features (7):** `Size`, `Weight`, `Sweetness`, `Crunchiness`, `Juiciness`, `Ripeness`, `Acidity`.
* **Top Performing Model:** **K-Nearest Neighbors (KNN)** achieving **87.50% - 90.25% test accuracy** and **94.36% ROC-AUC**, significantly outperforming Gaussian Naive Bayes (**74.00% accuracy**).
* **Interactive Web Dashboard:** Dark-mode UI with real-time sliders, instant dual-model prediction, and live analytical visualizations.

---

## 📊 Performance Benchmark Comparison

Evaluated on a 20% stratified hold-out test set (800 test samples):

| Metric | K-Nearest Neighbors (k=5) | Gaussian Naive Bayes | Advantage / Margin |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **87.50%** | 74.00% | **KNN (+13.50%)** |
| **Precision (Good)** | **86.80%** | 73.59% | **KNN (+13.21%)** |
| **Recall (Good)** | **88.53%** | 75.06% | **KNN (+13.47%)** |
| **F1-Score** | **87.65%** | 74.32% | **KNN (+13.33%)** |
| **ROC-AUC** | **94.36%** | 83.05% | **KNN (+11.31%)** |

---

## 🧠 Why Does KNN Outperform Naive Bayes?

1. **Violation of Feature Independence Assumption:**
   Naive Bayes applies Bayes' Theorem under the strict assumption that all input features are conditionally independent given the class label:
   $$P(X \mid C) = \prod_{i=1}^{n} P(x_i \mid C)$$
   In real fruits, physical attributes (such as sweetness, ripeness, juiciness, and size) have non-linear correlations and cross-dependencies.
2. **Non-Parametric Flexibility:**
   KNN is an instance-based, non-parametric algorithm that calculates Euclidean distances in multidimensional space:
   $$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
   It makes no assumptions about Gaussian distributions and forms local, non-linear decision boundaries that fit the fruit quality distribution.
3. **Importance of Feature Scaling (`StandardScaler`):**
   Because KNN relies strictly on geometric distance, unscaled features with larger variance dominate the calculation. Standardizing features ($z = \frac{x - \mu}{\sigma}$) boosted KNN accuracy from unscaled baseline to >90%.

---

## 🗂️ Project Directory Structure

```text
AiProject/
├── app.py                         # Interactive Flask Web Dashboard Server
├── train_and_evaluate.py          # Master model training, evaluation & plot generator
├── knn_classifier.py              # Clean, standalone KNN classifier script
├── naive_bayes_classifier.py      # Clean, standalone Gaussian Naive Bayes script
├── apple_quality_analysis.ipynb   # Complete Jupyter Notebook with step-by-step EDA & models
├── applequality.csv               # Apple quality dataset (4,000 samples)
├── requirements.txt               # Project Python dependencies
├── README.md                      # Comprehensive project documentation
├── VIVA_PRESENTATION_GUIDE.md     # Viva questions, answers, and presentation cheat sheet
├── models/                        # Serialized pre-trained model artifacts
│   ├── knn_model.pkl              # Trained KNN model
│   ├── nb_model.pkl               # Trained GaussianNB model
│   └── scaler.pkl                 # StandardScaler object
├── static/                        # Frontend static web assets
│   ├── style.css                  # Modern dark glassmorphic styling (Vanilla CSS)
│   ├── app.js                     # Interactive sliders and API controller
│   └── plots/                     # High-resolution benchmark figures
│       ├── model_comparison.png   # Accuracy, Precision, Recall comparison
│       ├── confusion_matrices.png # Side-by-side Confusion Matrices
│       ├── feature_correlations.png # Correlation heatmap with Quality
│       └── feature_distributions.png # Density distribution plots
└── templates/
    └── index.html                 # Semantic HTML5 dashboard template
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Installation

Clone or open the project folder in terminal, then install required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Train Models & Generate Plots

To re-train the models and regenerate all analytical charts:

```bash
python train_and_evaluate.py
```

### 3. Run Standalone Classifiers

To run KNN standalone:
```bash
python knn_classifier.py
```

To run Gaussian Naive Bayes standalone:
```bash
python naive_bayes_classifier.py
```

### 4. Launch the Interactive Web Application

Start the Flask application server:

```bash
python app.py
```

Then open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

---

## 💻 Web Application Features

1. **Live Predictor:** Adjust the 7 feature sliders or click instant presets (**"Crisp & Sweet"**, **"Overripe Soft"**, or **"Random Dataset Sample"**) to get real-time quality predictions.
2. **Model Duel:** View side-by-side classification outputs from both KNN and Naive Bayes, along with confidence percentages and agreement indicators.
3. **Model Benchmarks:** View interactive KPI cards, evaluation metric comparisons, and high-resolution confusion matrix charts.
4. **Dataset Insights:** Explore feature correlation heatmaps and density distributions.
5. **Viva & Theory Tab:** Comprehensive accordion guide containing answers to common oral examination and presentation questions.

---

## 📜 Academic Attribution

* Dataset: Kaggle Apple Quality Dataset
* Target: Agricultural Quality Control & AI Automated Inspection
