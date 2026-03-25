import pandas as pd
from core.preprocessing import split_features_target
import pytest

def test_split_features_target():
    data = {
        "a": [1, 2],
        "b": [3, 4],
        "target": [0, 1]
    }

    df = pd.DataFrame(data)
    X, y = split_features_target(df, "target")

    assert X.shape == (2, 2)
    assert y.shape == (2,)
    
    assert list(X.columns) == ["a", "b"]

    assert y.tolist() == [0, 1]

def test_split_features_target_invalid_target():
    data = {
        "a": [1, 2],
        "b": [3, 4]
    }

    df = pd.DataFrame(data)

    with pytest.raises(ValueError):
        split_features_target(df, "target")