import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OneHotEncoder
from backend.data_manager import DataManager
from backend.factory import LoaderFactory
import pandas as pd
import numpy as np

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

def test_random_forest_classifier_model_estimator(manager_setup):
    model, y_test, y_pred = manager_setup.ml.train(
        target = "target",
        task = "classification",
        model = "random_forest"
    )
    assert isinstance(model.named_steps['classifier'], RandomForestClassifier)
    assert len(y_test) == len(y_pred)
    assert len(y_test) > 0

def test_random_forest_regression_model_estimator(manager_setup):
    model, y_test, y_pred = manager_setup.ml.train(
        target = "B",
        task = "regression",
        model = "random_forest"
    )
    assert isinstance(model.named_steps['regressor'], RandomForestRegressor)
    assert len(y_test) == len(y_pred)
    assert len(y_test) > 0

def test_logistic_regression_model_estimator(manager_setup):
    model, y_test, y_pred = manager_setup.ml.train(
        target="target",
        task = "classification",
        model = "logistic_regression"
    )
    assert isinstance(model.named_steps['classifier'], LogisticRegression)
    assert len(y_test) == len(y_pred)
    assert len(y_test) > 0

def test_linear_regression_model_estimator(manager_setup):
    model, y_test, y_pred = manager_setup.ml.train(
        target="B",
        task = "regression",
        model = "linear_regression"
    )
    assert isinstance(model.named_steps['regressor'], LinearRegression)
    assert len(y_test) == len(y_pred)
    assert len(y_test) > 0

@pytest.fixture
def manager_setup_with_missing_values(tmp_path):
    df = pd.DataFrame({
                "age": [22, 30, None, 45, 28, 35, 40, 50, 26, 33],
                "city": ["Delhi", "Noida", "Delhi", np.nan, "Noida", "Delhi", "Agra", "Noida", "Agra", "Delhi"],
                "target": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"]
            })
    
    temp_file = tmp_path / "test_data_with_missing.csv"
    
    df.to_csv(temp_file, index=False)

    loader = LoaderFactory.create_loader(temp_file)
    manager = DataManager(loader)
    manager.load()
    return manager

def test_missing_values_handling(manager_setup_with_missing_values):
    model, y_test, y_pred = manager_setup_with_missing_values.ml.train(
        target="target",
        task="classification",
        model="random_forest"
    )
    assert isinstance(model.named_steps['classifier'], RandomForestClassifier)
    preprocessor = model.named_steps['preprocessor']
    numeric_transformer = preprocessor.named_transformers_['num']
    categorical_transformer = preprocessor.named_transformers_['cat']
    imputer_numeric = numeric_transformer.named_steps['imputer']
    statistics_numeric = imputer_numeric.statistics_
    imputer_categorical = categorical_transformer.named_steps['imputer']
    statistics_categorical_imputed = imputer_categorical.statistics_
    statistics_categorical_onehot = categorical_transformer.named_steps['onehot'].categories_
    assert isinstance(numeric_transformer.named_steps['imputer'], SimpleImputer)
    assert isinstance(categorical_transformer.named_steps['imputer'], SimpleImputer)
    assert isinstance(categorical_transformer.named_steps['onehot'], OneHotEncoder)
    assert np.isclose(statistics_numeric[0], 35.0)
    assert statistics_categorical_imputed[0] == 'Delhi'
    np.testing.assert_array_equal(statistics_categorical_onehot[0], ['Agra', 'Delhi', 'Noida'])
    assert len(y_test) == len(y_pred)
    assert len(y_test) > 0

def test_onehotencoder_handle_unknown():
    encoder = OneHotEncoder(handle_unknown='ignore')
    data = np.array(["Delhi", "Mumbai", "Noida"]).reshape(-1, 1)
    encoder.fit(data)
    new_data = np.array([["Delhi", "Noida", "Agra"]]).reshape(-1, 1)
    transformed_data = encoder.transform(new_data).toarray()
    expected_output = np.array([[1., 0., 0.],
                                [0., 0., 1.],
                                [0., 0., 0.]])
    np.testing.assert_array_equal(transformed_data, expected_output)