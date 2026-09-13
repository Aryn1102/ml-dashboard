import pandas as pd
import numpy as np
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from backend.data_manager import DataManager

class DataAnalyzer:
    def __init__(self, manager:"DataManager") -> None:
        self.manager = manager
    
    def _ensure_loaded(self) -> None:
        if self.manager.data is None:
            raise ValueError("Dataset not loaded. Call DataManager.load() first.")
    
    def shape(self) -> tuple[int, int]:
        self._ensure_loaded()
        return self.manager.data.shape
    
    def columns(self) -> pd.Index:
        self._ensure_loaded()
        return self.manager.data.columns

    def dtypes(self) -> pd.Series:
        self._ensure_loaded()
        return self.manager.data.dtypes
    
    def describe(self) -> pd.DataFrame:
        self._ensure_loaded()
        return self.manager.data.describe()

    def info(self) -> None:
        self._ensure_loaded()
        self.manager.data.info()
    
    def missing_values(self) -> pd.Series:
        self._ensure_loaded()
        return self.manager.data.isnull().sum()

    def duplicates(self) -> pd.DataFrame:
        self._ensure_loaded()
        return self.manager.data[self.manager.data.duplicated()]

    def unique_values(self, column:str) -> np.ndarray:
        self._ensure_loaded()
        if column in self.manager.data.columns:
            return self.manager.data[column].unique()
        else:
            raise ValueError(f"{column} not found.")

    def value_counts(self, column:str) -> pd.Series:
        self._ensure_loaded()
        if column in self.manager.data.columns:
            return self.manager.data[column].value_counts()
        else:
            raise ValueError(f"{column} not found.")

    def numeric_columns(self) -> list[str]:
        self._ensure_loaded()
        return self.manager.data.select_dtypes(include=['int64', 'float64']).columns.to_list()

    def categorical_columns(self) -> list[str]:
        self._ensure_loaded()
        return self.manager.data.select_dtypes(include=['object', 'category']).columns.to_list()