from backend.base_loader import DatasetLoader
import  pandas as pd

class JsonLoader(DatasetLoader):
    def load(self) -> pd.DataFrame:
        return pd.read_json(self.source)