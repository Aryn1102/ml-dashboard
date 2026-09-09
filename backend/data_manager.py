from backend.base_loader import DatasetLoader
import pandas as pd
from backend.data_analyzer import DataAnalyzer
from backend.data_cleaner import DataCleaner
class DataManager:
    def __init__(self, loader:DatasetLoader):
        self.loader = loader
        self.data: pd.DataFrame | None = None
        self._analyzer = None
        self._cleaner = None

    def load(self) -> pd.DataFrame:
        self.data = self.loader.load()
        return self.data

    def _ensure_loaded(self):
        if self.data is None:
            raise ValueError("Dataset not Loaded.")

    def save(self, output_path: str) -> None:
        self._ensure_loaded()
        self.data.to_csv(output_path, index=False)
    
    @property
    def analyzer(self) -> DataAnalyzer:
        if self._analyzer is None:
            self._analyzer = DataAnalyzer(self)
        return self._analyzer
    
    @property
    def cleaner(self) -> DataCleaner:
        if self._cleaner is None:
            self._cleaner = DataCleaner(self)
        return self._cleaner