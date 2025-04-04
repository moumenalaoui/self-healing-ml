# Self-Healing Machine Learning - MLOps

A lightweight framework to add "self-healing" capabilities to your machine learning models. This wrapper automatically monitors model performance and retrains when performance drops below a defined threshold.

Ideal for scenarios where your model faces **concept drift**, data evolution, or you just want a plug-and-play way to keep performance in check.

---

## Features

- **Performance Monitoring** – Automatically evaluates your model on validation data
- **Automatic Retraining** – Retrains your model when accuracy (or other metrics) drops
- **Customizable Metrics** – Use any metric function (e.g. accuracy, F1, RMSE)
- **Works with Any Model** – Compatible with most `sklearn`-style models

---

## Installation

```bash
git clone https://github.com/moumenalaoui/self-healing-ml.git
cd self-healing-ml
pip install -r requirements.txt
```
You may also want to create a virtual environment before installing:

```bash
python -m venv venv
source venv/bin/activate
```

## Basic Usage:

```python
from self_heal import SelfHealingModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load sample data
X, y = load_iris(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model and self-healing wrapper
base_model = RandomForestClassifier()
healing_model = SelfHealingModel(base_model, metric=accuracy_score, threshold=0.9)

# Train on training set
healing_model.train(X_train, y_train)

# Monitor and retrain if needed
healing_model.monitor_and_retrain(X_val, y_val)
```

## Class Overview:

SelfHealingModel(model, metric, threshold)

    model: Any ML model with .fit() and .predict() methods

    metric: A function like accuracy_score(y_true, y_pred)

    threshold: Float between 0–1 to decide when to retrain

Methods:

    .train(X, y): Trains the model

    .evaluate(X, y): Returns performance score

    .monitor_and_retrain(X_val, y_val): Retrains if performance < threshold

  ## Project Structure:
  
```bash
self-healing-ml/
├── src/
│   └── self_heal.py     # Core self-healing logic
├── examples/            # Example usage scripts (optional)
├── requirements.txt
└── README.md
```

## License:
This project is licensed under the MIT License. See the LICENSE file for more details.
