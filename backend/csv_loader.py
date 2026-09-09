from backend.base_loader import DatasetLoader
import pandas as pd
class CSVLoader(DatasetLoader):
    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.source)