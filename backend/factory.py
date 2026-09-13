from typing import IO

from backend.csv_loader import CSVLoader
from backend.excel_loader import ExcelLoader
from backend.json_loader import JSONLoader
from backend.base_loader import DatasetLoader
from pathlib import Path

class LoaderFactory:
    LOADERS={
        ".csv": CSVLoader,
        ".xlsx": ExcelLoader,
        ".xls": ExcelLoader,
        ".json": JSONLoader,
    }

    @classmethod
    def create_loader(cls, source: str | Path | IO) -> DatasetLoader:
        if isinstance(source, (str, Path)):
            filename = str(source)
        elif hasattr(source, "name"):
            filename = source.name
        else:
            raise ValueError("Unsupported File")
        extension = Path(filename).suffix.lower()
        loader_class = cls.LOADERS.get(extension)
        if loader_class is None:
            raise ValueError(f"Unsupported file type: {extension}")
        return loader_class(source)
