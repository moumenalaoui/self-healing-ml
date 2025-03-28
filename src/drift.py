import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset

# File paths
TRAIN_PATH = "data/train.csv"
NEW_DATA_PATH = "data/new_data.csv"

# Column names for consistency
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

def run_drift_detection():
    df_train = pd.read_csv(TRAIN_PATH, header=None, names=COLUMNS)
    df_new = pd.read_csv(NEW_DATA_PATH, header=None, names=COLUMNS)

    # Drop target column for drift detection
    df_train = df_train.drop(columns=["income"])
    df_new = df_new.drop(columns=["income"])

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_train, current_data=df_new)

    report.save_html("drift_report.html")
    print("Drift report saved as drift_report.html")

if __name__ == "__main__":
    run_drift_detection()
