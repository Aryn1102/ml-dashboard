import pandas as pd
from backend.data_manager import DataManager
from backend.factory import LoaderFactory
import pytest
def test_distributions(tmp_path):
    df = pd.DataFrame({
        "A": [1, 2, 2, 3, 4, 5, 6, 7, 8],
        "B": [10, 20, 30, 40, 50, 60, 70, 80, 90],
        "Category": ["A", "B", "A", "B", "A", "B", "A", "B", "A"]
    })
    temp_file = tmp_path / "test_data.csv"
    
    df.to_csv(temp_file, index=False)

    loader = LoaderFactory.create_loader(temp_file)
    manager = DataManager(loader)
    manager.load()

    distributions = manager.eda.distributions("A")
    assert distributions["mean"] == pytest.approx(4.222, abs = 0.001)
    assert distributions["median"] == 4
    assert distributions["mode"] == [2]
    assert distributions["min"] == 1
    assert distributions["max"] == 8
    assert distributions["quantiles"]["Q1"] == 2
    assert distributions["quantiles"]["Q3"] == 6
    assert distributions["skew"] == pytest.approx(0.26832344799098923)
    assert distributions["kurtosis"] == pytest.approx(-1.293912132063935)