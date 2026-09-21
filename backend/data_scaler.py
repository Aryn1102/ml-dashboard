from typing import TYPE_CHECKING
if TYPE_CHECKING:    
    from backend.data_manager import DataManager
from sklearn.preprocessing import StandardScaler

class DataScaler:
    def __init__(self, manager: "DataManager") -> None:
        self.manager = manager

    def _ensure_loaded(self) -> None:
        if self.manager.data is None:
            raise ValueError("Dataset not loaded. Call Datamanager.load() first")

    def standard_scaler(self) -> None:
        self._ensure_loaded()
        num_col_name = self.manager.analyzer.numeric_columns()
        num_col = self.manager.data[num_col_name]
        scaler = StandardScaler()
        self.manager.data[num_col_name] = scaler.fit_transform(num_col)