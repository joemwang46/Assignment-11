import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

def read_features(file_path: str) -> list:
    with open(file_path, 'r') as file:
        output = json.load(file)

    features = output.get("features", [])
    return features

def read_model_options(file_path: str) -> list:
    with open(file_path, 'r') as file:
        output = json.load(file)

    model_options = list(output.keys())
    return model_options

def read_model_params(file_path: str, model: str) -> dict:
    with open(file_path, 'r') as file:
        output = json.load(file)

    model_params = output.get(model, {})
    return model_params

def initialize_model(model: str, params: dict):
    if model == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)
    elif model == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    elif model == "XGBClassifier":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model}")
    
def evaluate_model(model_name, y_test, y_pred):
    print(f"\n==== {model_name} Evaluation ====")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    plt.title(f"{model_name} - Confusion Matrix")
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.show()

def evaluate_feature_importance(model, model_name, feature_names):

    if model_name == "LogisticRegression":
        coefs = model.coef_[0]
        importance = sorted(zip(feature_names, abs(coefs)), key=lambda x: x[1], reverse=True)
        print("\nTop Features (Logistic Regression):")
        for f, w in importance:
            print(f"{f:20s}  {w:.4f}")
        return importance
    
    elif model_name == "RandomForestClassifier":
        importances = model.feature_importances_
        ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

        print("\nTop Features (Random Forest):")
        for f, w in ranked:
            print(f"{f:20s}  {w:.4f}")

        return ranked

    elif model_name == "XGBClassifier":
        scores = model.feature_importances_
        ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)

        print("\nTop Features (XGBoost):")
        for f, w in ranked:
            print(f"{f:20s}  {w:.4f}")

        return ranked
    
    else:
        raise ValueError(f"Unsupported model type: {model}")