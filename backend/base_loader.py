import pandas as pd
from abc import ABC, abstractmethod

class DatasetLoader(ABC):
    def __init__(self, source):
        self.source = source
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass