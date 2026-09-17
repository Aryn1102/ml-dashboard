from backend.data_manager import DataManager
from backend.factory import LoaderFactory

def test_one_hot_encode():
    loader = LoaderFactory.create_loader("train.csv")
    manager = DataManager(loader)
    manager.load()

    before_cat_data = manager.analyzer.categorical_columns()
    before_num_data = manager.analyzer.numeric_columns()
    row_count, _ = manager.analyzer.shape()
    manager.encoder.one_hot_encode()
    
    assert row_count == manager.data.shape[0]
    for col in before_cat_data:
        assert col not in manager.data.columns
    for column in before_num_data:
        assert column in manager.data.columns
    assert manager.data.shape[1] > len(before_num_data)