import pandas as pd
import tempfile
import os
import pytest
from core.ingestion import load_data


def test_load_data_returns_dataframe():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("a,b,target\n1,2,0\n3,4,1\n")
        temp_path = f.name

    df = load_data(temp_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)

    os.remove(temp_path)


def test_load_data_drops_empty_rows():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("a,b,target\n1,2,0\n,,\n3,4,1\n")
        temp_path = f.name

    df = load_data(temp_path)

    assert df.shape == (2, 3)

    os.remove(temp_path)


def test_load_data_empty_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        temp_path = f.name

    with pytest.raises(Exception):
        load_data(temp_path)

    os.remove(temp_path)