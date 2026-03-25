import pandas as pd
from core.model import detect_problem_type


def test_classification_detection():
    y = pd.Series([0, 1, 0, 1])
    assert detect_problem_type(y) == "classification"


def test_regression_detection():
    y = pd.Series([10, 20, 30, 40])
    assert detect_problem_type(y) == "regression"