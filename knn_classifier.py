"""
Apple Quality Classification - K-Nearest Neighbors (KNN)
Clean, standalone implementation with relative dataset path and detailed metrics.
"""

import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def run_knn():
    # 1. Dataset path (uses local folder)
    csv_path = os.path.join(os.path.dirname(__file__), "applequality.csv")
    if not os.path.exists(csv_path):
        csv_path = "applequality.csv"

    print(f"Loading data from: {csv_path}")
    data = pd.read_csv(csv_path)

    # 2. Data Cleaning
    if "A_id" in data.columns:
        data = data.drop("A_id", axis=1)

    # 3. Features & Target
    X = data.drop("Quality", axis=1)  # Features
    y = data["Quality"]              # Target class ('good' or 'bad')

    # 4. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Feature Scaling (StandardScaler) - Essential for distance-based KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Initialize and Train KNN
    k = 5  # Number of neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    # 7. Predictions & Evaluation
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 45)
    print(" K-Nearest Neighbors (KNN) Model Results ")
    print("=" * 45)
    print(f"Neighbors (k) : {k}")
    print(f"Accuracy Score: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=["good", "bad"]))
    print("=" * 45)

    return knn, scaler, accuracy


if __name__ == "__main__":
    run_knn()
