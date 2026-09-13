from backend.factory import LoaderFactory
from backend.data_manager import DataManager    

def test_data_cleaner():
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
