import pandas as pd
import numpy as np

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

def simulate_valid_drift():
    df = pd.read_csv("data/train.csv", header=None, names=COLUMNS)

    # Introduce **variance** with shifted distributions
    df["age"] = np.random.randint(60, 80, size=len(df))  # Older
    df["education"] = np.random.choice(["Preschool", "1st-4th", "Doctorate"], size=len(df))
    df["workclass"] = np.random.choice(["Never-worked", "Private", "State-gov"], size=len(df))
    df["hours_per_week"] = np.random.normal(loc=15, scale=5, size=len(df)).astype(int)
    df["native_country"] = np.random.choice(["India", "Mexico", "United-States"], size=len(df))

    df.to_csv("data/new_data.csv", index=False, header=False)
    print("Simulated diverse drifted dataset saved to data/new_data.csv")

if __name__ == "__main__":
    simulate_valid_drift()
