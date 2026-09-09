from backend.base_loader import DatasetLoader
import pandas as pd

class ExcelLoader(DatasetLoader):
    def load(self) -> pd.DataFrame:
        return pd.read_excel(self.source)