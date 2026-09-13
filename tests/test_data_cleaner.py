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
    

if __name__ == "__main__":
    test_data_cleaner()