import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import mlflow

# File paths
DATA_PATH = "data/train.csv"
MODEL_PATH = "models/model.pkl"

# Column names for Adult dataset (added here to fix missing headers)
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

def load_data(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)

    # Drop missing values
    df = df.replace("?", np.nan).dropna()

    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

def train_model(df):
    X = df.drop("income", axis=1)
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Model Accuracy: {acc:.4f}")

    return clf, acc

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("self_healing_ml")

    with mlflow.start_run():
        df, encoders = load_data(DATA_PATH)
        model, acc = train_model(df)
        save_model(model, MODEL_PATH)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_artifact(MODEL_PATH)

if __name__ == "__main__":
    main()
