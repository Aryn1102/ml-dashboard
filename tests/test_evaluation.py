import numpy as np
from core.evaluation import evaluate_model


def test_classification_evaluation():
    y_test = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 0, 1])

    metrics = evaluate_model("classification", y_test, preds)

    assert "accuracy" in metrics
    assert isinstance(metrics["accuracy"], float)


def test_regression_evaluation():
    y_test = np.array([10.0, 20.0, 30.0])
    preds = np.array([12.0, 18.0, 29.0])

    metrics = evaluate_model("regression", y_test, preds)

    assert "r2_score" in metrics
    assert isinstance(metrics["r2_score"], float)