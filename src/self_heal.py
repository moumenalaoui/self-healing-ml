import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import mlflow

# --- Config ---
TRAIN_PATH = "data/train.csv"
NEW_DATA_PATH = "data/new_data.csv"
MODEL_PATH = "models/model.pkl"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

def load_data(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df = df.replace("?", np.nan).dropna()

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

def detect_drift(train_df, new_df, threshold=3):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train_df.drop(columns=["income"]),
               current_data=new_df.drop(columns=["income"]))
    results = report.as_dict()

    drifted_features = []

    try:
        for metric in results["metrics"]:
            if metric["metric"] == "DataDriftTable":
                for feature, detail in metric["result"]["drift_by_columns"].items():
                    if detail.get("drift_detected", False):
                        drifted_features.append(feature)
                break
    except Exception as e:
        print(f"Error while parsing drift results: {e}")
        return False

    print(f"Drifted features: {drifted_features}")
    return len(drifted_features) >= threshold

def train_and_evaluate(df):
    X = df.drop("income", axis=1)
    y = df["income"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return clf, acc

def load_existing_model_and_score(df):
    if not os.path.exists(MODEL_PATH):
        return 0.0

    model = joblib.load(MODEL_PATH)
    X = df.drop("income", axis=1)
    y = df["income"]
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    return acc

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("self_healing_ml")

    with mlflow.start_run():
        df_train, _ = load_data(TRAIN_PATH)
        df_new, _ = load_data(NEW_DATA_PATH)

        print("Checking for drift...")
        drift_detected = detect_drift(df_train, df_new)

        if not drift_detected:
            print("No significant drift detected. Keeping current model.")
            return

        print("Drift detected. Retraining model...")
        new_model, new_acc = train_and_evaluate(df_new)
        old_acc = load_existing_model_and_score(df_new)

        print(f"New Model Accuracy: {new_acc:.4f}")
        print(f"Old Model Accuracy on new data: {old_acc:.4f}")

        if new_acc > old_acc:
            joblib.dump(new_model, MODEL_PATH)
            mlflow.log_param("action", "model_retrained_and_replaced")
            print("New model outperformed old one. Replaced.")
        else:
            mlflow.log_param("action", "model_retrained_but_not_replaced")
            print("New model did not outperform old one. Keeping existing.")

        mlflow.log_metric("new_accuracy", new_acc)
        mlflow.log_metric("old_accuracy", old_acc)

if __name__ == "__main__":
    main()
