import pytest
from sklearn.ensemble import RandomForestClassifier

from backend.data_manager import DataManager
from backend.factory import LoaderFactory
import pandas as pd

@pytest.fixture
def manager_setup(tmp_path):
    df = pd.DataFrame({
                "A": [1, 2, 2, 3, 4, 5, 6, 7, 8],
                "B": [10, 20, 30, 40, 50, 60, 70, 80, 90],
                "target": ["A", "B", "A", "B", "A", "B", "A", "B", "A"]
            })
    
    temp_file = tmp_path / "test_data.csv"
    
    df.to_csv(temp_file, index=False)

    loader = LoaderFactory.create_loader(temp_file)
    manager = DataManager(loader)
    manager.load()
    return manager

def test_invalid_target(manager_setup):
    with pytest.raises(ValueError):
        manager_setup.ml.train(
            target = "category",
            task = "classification",
            model = "random_forest"
        )

def test_invalid_task(manager_setup):
    with pytest.raises(ValueError):
        manager_setup.ml.train(
            target = "target",
            task = "clustering",
            model = "random_forest"
        )

def test_invalid_model(manager_setup):
    with pytest.raises(ValueError):
        manager_setup.ml.train(
            target = "target",
            task = "classification",
            model = "svm"
        )

def test_valid_model_task(manager_setup):
    manager_setup.ml.train(
        target = "target",
        task = "classification",
        model = "random_forest"
    )

def test_model_task_mismatch(manager_setup):
    with pytest.raises(ValueError):
        manager_setup.ml.train(
            target = "target",
            task = "regression",
            model = "logistic_regression"
        )

def test_model_estimator(manager_setup):
    model, y_test, y_pred = manager_setup.ml.train(
        target = "target",
        task = "classification",
        model = "random_forest"
    )
    assert isinstance(model, RandomForestClassifier)
    assert len(y_test) == len(y_pred)
    assert len(y_test) > 0
