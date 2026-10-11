from backend.preprocessing.encoder import DataEncoder
import pytest
import pandas as pd

@pytest.fixture
def encoder():
    return DataEncoder()

@pytest.fixture
def X():
    return pd.DataFrame({
        'color': ['red', 'green', 'blue'],
        'size': ['S', 'M', 'L']
    },
    index=[10, 20, 30]
    )

def test_fit(encoder, X):
    encoder.fit(X)
    assert encoder._is_fitted == True

def test_before_transform(encoder, X):
    with pytest.raises(ValueError):
        encoder.transform(X)

def test_transform(encoder, X):
    encoder.fit(X)
    column = encoder.encoder.get_feature_names_out()
    result = encoder.transform(X)
    assert list(result.columns) == list(column)

def test_fit_transform(encoder, X):
    result = encoder.fit_transform(X)
    column = encoder.encoder.get_feature_names_out()
    assert list(result.columns) == list(column)

def test_index(encoder, X):
    encoder.fit(X)
    result = encoder.transform(X)
    assert list(result.index) == list(X.index)

def test_handle_ignore(encoder):
    X_train = pd.DataFrame({
        'color': ['red', 'green', 'blue']
    })
    X_test = pd.DataFrame({
        'color': ['red', 'yellow']
    })
    encoder.fit(X_train)
    result = encoder.transform(X_test)
    assert result.shape[0] == X_test.shape[0]
    assert 'color_yellow' not in result.columns
    assert result['color_red'].iloc[1] == 0
    assert result['color_blue'].iloc[1] == 0
    assert result['color_green'].iloc[1] == 0