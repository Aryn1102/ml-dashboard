from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.data_manager import DataManager
from backend.data_analyzer import DataAnalyzer  
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class DataEncoder:
    def __init__(self, manager: DataManager) -> None:
        self.manager = manager

    def _ensure_loaded(self) -> None:
            if self.manager.data is None:
                raise ValueError("Dataset not loaded. Call DataManager.load() first.")

    def one_hot_encode(self) -> None:
        self._ensure_loaded()
        data = self.manager.data[self.manager.analyzer.categorical_columns()]
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_data = encoder.fit_transform(data)
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
        self.manager.data = pd.concat([self.manager.data.drop(columns=data.columns), encoded_df], axis=1)