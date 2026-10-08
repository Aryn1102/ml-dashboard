import pytest
from backend.preprocessing.data_scaler import DataScaler

@pytest.fixture
def scaler():
    return DataScaler()

@pytest.fixture
def X():
    return [[1, 2], [3, 4]]

def test_fit(scaler, X):
    scaler.fit(X)
    assert scaler._is_fitted == True

def test_transform(scaler, X):
    scaler.fit(X)
    result = scaler.transform(X)
    assert result.mean(axis = 0) == pytest.approx(0)
    assert result.std(axis = 0) == pytest.approx(1)

def test_fit_transform(scaler, X):
    result = scaler.fit_transform(X)
    assert scaler._is_fitted == True
    assert result.mean(axis = 0) == pytest.approx(0)
    assert result.std(axis = 0) == pytest.approx(1)

def test_transform_before_fit(scaler, X):
    with pytest.raises(ValueError):
        scaler.transform(X)