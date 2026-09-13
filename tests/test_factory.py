import pytest
from backend.factory import LoaderFactory
from backend.csv_loader import CSVLoader
from backend.excel_loader import ExcelLoader
from backend.json_loader import JSONLoader

def test_create_loader_with_csv():
    loader = LoaderFactory.create_loader("train.csv")
    assert isinstance(loader, CSVLoader)

def test_create_loader_with_excel():
    loader = LoaderFactory.create_loader("train.xlsx")
    assert isinstance(loader, ExcelLoader)

def test_create_loader_with_unsupported_file():
    with pytest.raises(ValueError):
        LoaderFactory.create_loader("train.txt")
        
def test_create_loader_with_json():
    loader = LoaderFactory.create_loader("train.json")
    assert isinstance(loader, JSONLoader)