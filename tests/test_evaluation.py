import pytest
from backend.evaluate import Evaluation

@pytest.fixture
def evaluation_data():
    return {
        "classification": {
            "y_test": [0, 0, 1, 1],
            "y_pred": [0, 1, 1, 1],
            "task": "classification",
        },
        "regression": {
            "y_test": [2, 4, 6, 8],
            "y_pred": [2, 5, 5, 8],
            "task": "regression",
        },
    }

def test_valid_classification(evaluation_data):
    data = evaluation_data["classification"]
    result = Evaluation().evaluate(
        data["y_test"], 
        data["y_pred"],
        data["task"]
    )

    assert result["accuracy"] == 0.75
    assert result["precision"] == pytest.approx(0.8333, rel=1e-3)
    assert result["recall"] == 0.75
    assert result["f1"] == pytest.approx(0.7333, rel=1e-3)

    expected_matrix = [
        [1, 1],
        [0, 2]
    ]

    assert result["confusion_matrix"].tolist() == expected_matrix

def test_valid_regression(evaluation_data):
    data = evaluation_data["regression"]
    result = Evaluation().evaluate(
        data["y_test"], 
        data["y_pred"],
        data["task"]
    )

    assert result["rmse"] == pytest.approx(0.70710678)
    assert result["mae"] == 0.5
    assert result["r2"] == pytest.approx(0.9)

def test_invalid_task(evaluation_data):
    data = evaluation_data["classification"]

    with pytest.raises(ValueError):
        Evaluation().evaluate(
            data["y_test"],
            data["y_pred"],
            "clustering"
        )

def test_mismatched_lengths(evaluation_data):
    data = evaluation_data["classification"]

    with pytest.raises(ValueError):
        Evaluation().evaluate(
            data["y_test"],
            data["y_pred"][:-1],
            data["task"]
        )