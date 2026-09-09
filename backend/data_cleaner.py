from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backend.data_manager import DataManager
import pandas as pd
class DataCleaner:

    def __init__(self, manager:"DataManager") -> None:
        self.manager = manager
    
    def _ensure_loaded(self) -> None:
        if self.manager.data is None:
            raise ValueError("Dataset not Loaded. Call DataManager.load() first.")
    
    def remove_duplicates(self, subset: list[str] | None = None, keep: str = 'first', ignore_index: bool = True) -> None:
        self._ensure_loaded()
        self.manager.data = self.manager.data.drop_duplicates(subset=subset,keep=keep,ignore_index=ignore_index)
    
    def fill_missing(self, mapping:dict) -> None:
        self._ensure_loaded()
        self.manager.data = self.manager.data.fillna(mapping)
    
    def standard_date_time_format(self, columns: list[str]) -> None:
        self._ensure_loaded()
        for col in columns:
            if col in self.manager.data.columns:
                self.manager.data[col] = pd.to_datetime(self.manager.data[col], errors='coerce')
            else:
                raise ValueError(f"{col} not found.")
    
    def standardize_text(self, columns: list[str]) -> None:
        self._ensure_loaded()
        for col in columns:
            if col in self.manager.data.columns:
                self.manager.data[col] = self.manager.data[col].astype(str).str.lower().str.strip()
            else:
                raise ValueError(f"{col} not found.")
    
    def categorical_mapping(self, columns: list[str], mapping_dict: dict) -> None:
        self._ensure_loaded()
        for col in columns:
            if col in self.manager.data.columns:
                self.manager.data[col] = self.manager.data[col].replace(mapping_dict)
            else:
                raise ValueError(f"{col} not found.")
    
    def replace_nan(self) -> None:
        self._ensure_loaded()
        self.manager.data.replace([
    "N/A",
    "NA",
    "null",
    "None",
    "missing",
    "unknown",
    "-",
    ""
], pd.NA, inplace=True)