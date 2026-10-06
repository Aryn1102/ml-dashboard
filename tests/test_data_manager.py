from backend.evaluate import Evaluation
import pytest
from backend.factory import LoaderFactory
from backend.data_manager import DataManager
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

def test_evaluation_property(manager_setup):
    evaluation = manager_setup.evaluation

    assert isinstance(evaluation, Evaluation)