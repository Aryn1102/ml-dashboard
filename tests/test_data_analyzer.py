from backend.factory import LoaderFactory
from backend.data_manager import DataManager

def test_shape():
    loader = LoaderFactory.create_loader("train.csv")
    manager = DataManager(loader)
    manager.load()

    shape = manager.analyzer.shape()

    assert shape == manager.data.shape

def test_columns():
    Loader = LoaderFactory.create_loader("train.csv")
    manager = DataManager(Loader)
    manager.load()

    columns = manager.analyzer.columns()

    assert columns.equals(manager.data.columns)

def test_dtypes():
    Loader = LoaderFactory.create_loader("train.csv")
    manager = DataManager(Loader)
    manager.load()

    dtypes = manager.analyzer.dtypes()

    assert dtypes.equals(manager.data.dtypes)

def test_missing_values():
    Loader = LoaderFactory.create_loader("train.csv")
    manager = DataManager(Loader)
    manager.load()

    missing_values = manager.analyzer.missing_values()

    assert missing_values.equals(manager.data.isnull().sum())

def test_duplicates():
    Loader = LoaderFactory.create_loader("train.csv")
    manager = DataManager(Loader)
    manager.load()

    duplicates = manager.analyzer.duplicates()

    assert duplicates.equals(manager.data[manager.data.duplicated()])

def test_unique_values():
    Loader = LoaderFactory.create_loader("train.csv")
    manager = DataManager(Loader)
    manager.load()

    unique_values = manager.analyzer.unique_values("sex")

    assert (unique_values == manager.data["sex"].unique()).all()

def test_value_counts():
    Loader = LoaderFactory.create_loader("train.csv")
    manager = DataManager(Loader)
    manager.load()

    value_counts = manager.analyzer.value_counts("sex")

    assert value_counts.equals(manager.data["sex"].value_counts())

def test_numeric_columns():
    Loader = LoaderFactory.create_loader("train.csv")
    manager = DataManager(Loader)
    manager.load()

    numeric_columns = manager.analyzer.numeric_columns()

    assert numeric_columns == manager.data.select_dtypes(include = ["int64", "float64"]).columns.to_list()

def test_categorical_columns():
    Loader = LoaderFactory.create_loader("train.csv")
    manager = DataManager(Loader)
    manager.load()

    categorical_columns = manager.analyzer.categorical_columns()

    assert categorical_columns == manager.data.select_dtypes(include = ["object", "category"]).columns.to_list()