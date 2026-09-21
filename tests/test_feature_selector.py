from backend.data_manager import DataManager
from backend.factory import LoaderFactory
import pandas as pd

def test_corr_pairs(tmp_path):
    df = pd.DataFrame({
    "A": [1, 2, 3, 4, 5],
    "B": [2, 4, 6, 8, 10],
    "C": [10, 8, 6, 4, 2],
    "D": [3, 1, 5, 2, 4]
    })

    temp_file = tmp_path / "test_data.csv"

    df.to_csv(temp_file, index=False)
    
    loader = LoaderFactory.create_loader(temp_file)
    manager = DataManager(loader)
    manager.load()

    corr_pairs = manager.feature_selector.correlation_pairs(threshold=0.8)

    assert corr_pairs[0] == ("A", "B", 1.0)
    assert corr_pairs[1] == ("A", "C", -1.0)