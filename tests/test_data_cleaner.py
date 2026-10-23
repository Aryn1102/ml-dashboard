from backend.factory import LoaderFactory
from backend.data_manager import DataManager    
import pandas as pd

def test_normalize_missing_values():
    loader = LoaderFactory.create_loader("train.csv")
    manager = DataManager(loader)
    manager.load()

    
    unknown_count = manager.data["deck"].value_counts()["unknown"]
    missing_before = manager.data["deck"].isna().sum()
    manager.cleaner.normalize_missing_values()
    missing_after = manager.data["deck"].isna().sum()
    assert "unknown" not in manager.data["deck"]
    assert missing_after- missing_before == unknown_count
    

def test_remove_duplicates():
    loader = LoaderFactory.create_loader("train.csv")
    manager = DataManager(loader)
    manager.load()

    row_before = manager.analyzer.shape()
    manager.cleaner.remove_duplicates()
    duplicates_after = manager.analyzer.duplicates()
    row_after = manager.analyzer.shape()
    assert row_after[0] < row_before[0] and duplicates_after.empty 

def test_missing_values():
    loader = LoaderFactory.create_loader("train.csv")
    manager = DataManager(loader)
    manager.load()
    missing_indices = manager.data.index[manager.data["age"].isna()]
    manager.cleaner.fill_missing({"age": 30})
    after_filling = manager.analyzer.missing_values()
    assert after_filling["age"] == 0
    assert all(manager.data.loc[missing_indices, "age"] == 30)

def test_standardize_text_preserves_missing():
    loader = LoaderFactory.create_loader("train.csv")
    manager = DataManager(loader)
    manager.load()

    manager.data["test_text"] = pd.NA
    manager.data.loc[manager.data.index[:3], "test_text"] = [
        " HELLO ", None, " WORLD "
        ]

    manager.cleaner.standardize_text(["test_text"])

    assert manager.data["test_text"].iloc[0] == "hello"
    assert pd.isna(manager.data["test_text"].iloc[1])
    assert manager.data["test_text"].iloc[2] == "world"

if __name__ == "__main__":
    test_normalize_missing_values()
    test_remove_duplicates()
    test_missing_values()
    test_standardize_text_preserves_missing()