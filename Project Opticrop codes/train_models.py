import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_compare():
    print("Starting Machine Learning Training Pipeline...")
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("static/images", exist_ok=True)

    # 1. Load Crop Recommendation Dataset
    df_rec = pd.read_csv("datasets/crop_recommendation_dataset.csv")
    X = df_rec.drop("label", axis=1)
    y = df_rec["label"]

    # Encode crop labels
    le_crop = LabelEncoder()
    y_encoded = le_crop.fit_transform(y)

    # Split into train & test
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Scale variables
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize classifiers
    models = {
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Evaluate models
    comparison = {}
    best_acc = 0
    best_model_name = ""
    best_model = None

    for name, clf in models.items():
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        rep = classification_report(y_test, preds, target_names=le_crop.classes_, output_dict=True)
        
        comparison[name] = {
            "accuracy": acc,
            "precision": rep["weighted avg"]["precision"],
            "recall": rep["weighted avg"]["recall"],
            "f1": rep["weighted avg"]["f1-score"]
        }
        print(f"Model: {name} | Test Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = clf

    print(f"\nWinning Model: {best_model_name} with {best_acc * 100:.2f}% Accuracy!")

    # 2. Save best models and encoders
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le_crop, f)
    print("Serialized best classifier, scaler, and label encoder.")

    # 3. K-Means Soil Clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    # Fit on all scaled soil features to cluster them
    X_all_scaled = scaler.transform(X)
    kmeans.fit(X_all_scaled)
    with open("models/kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    print("Serialized K-Means clustering model.")

    # 4. Train Yield Prediction Regressor
    df_yield = pd.read_csv("datasets/crop_yield_dataset.csv")
    
    # Multi-label categorical encodings
    le_location = LabelEncoder()
    df_yield["location_encoded"] = le_location.fit_transform(df_yield["location"])

    le_season = LabelEncoder()
    df_yield["season_encoded"] = le_season.fit_transform(df_yield["season"])

    # We map crop label as well
    df_yield["crop_encoded"] = le_crop.transform(df_yield["crop"])

    X_yield = df_yield[["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "crop_encoded", "location_encoded", "season_encoded"]]
    y_yield = df_yield["yield"]

    yield_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    yield_regressor.fit(X_yield, y_yield)

    with open("models/yield_model.pkl", "wb") as f:
        pickle.dump(yield_regressor, f)
    with open("models/yield_encoders.pkl", "wb") as f:
        pickle.dump({
            "location": le_location,
            "season": le_season
        }, f)
    print("Serialized Random Forest Yield regressor and categorical encoders.")

    # 5. Export charts to static/images for Flask dashboard
    # Chart A: Model Accuracy Comparisons
    plt.figure(figsize=(8, 4))
    names_list = list(comparison.keys())
    acc_list = [comparison[x]["accuracy"] * 100 for x in names_list]
    sns.barplot(x=acc_list, y=names_list, palette="viridis")
    plt.title("Model Accuracy Comparison (%)")
    plt.xlim(50, 105)
    plt.xlabel("Accuracy Score")
    plt.tight_layout()
    plt.savefig("static/images/model_comparison.png", dpi=150)
    plt.close()

    # Chart B: Confusion Matrix of best model
    best_preds = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=le_crop.classes_, yticklabels=le_crop.classes_)
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("static/images/confusion_matrix.png", dpi=150)
    plt.close()

    # Chart C: Feature Importances (Random Forest specific)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        features = X.columns
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(8, 4))
        sns.barplot(x=[importances[i] for i in indices], y=[features[i] for i in indices], palette="rocket")
        plt.title("Random Forest Gini Feature Importances")
        plt.xlabel("Relative Importance Score")
        plt.tight_layout()
        plt.savefig("static/images/feature_importance.png", dpi=150)
        plt.close()

    # Save metadata dictionary
    metadata = {
        "comparison": comparison,
        "best_model_name": best_model_name,
        "best_accuracy": best_acc
    }
    with open("models/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("Machine Learning Pipeline execution complete! Models saved in models/")

if __name__ == "__main__":
    train_and_compare()
