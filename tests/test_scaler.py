from backend.data_manager import DataManager
from backend.factory import LoaderFactory
import pytest

def test_Standard_Scaler():
    loader = LoaderFactory.create_loader("train.csv")
    manager = DataManager(loader)
    manager.load()

    before_cat_col = manager.analyzer.categorical_columns()
    before_rows, _ = manager.analyzer.shape()
    before_col_count = len(manager.analyzer.columns())
    before_num_col_name = manager.analyzer.numeric_columns()
    manager.scaler.standard_scaler()
    assert manager.data.shape == (before_rows, before_col_count)
    assert before_num_col_name == manager.data.select_dtypes(include=['int64', 'float64']).columns.to_list()
    assert before_cat_col == manager.data.select_dtypes(include=['object', 'category']).columns.to_list()
    for col in before_num_col_name:
        assert manager.data[col].mean() == pytest.approx(0)
        assert manager.data[col].std(ddof=0) == pytest.approx(1)