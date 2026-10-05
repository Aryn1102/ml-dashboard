from backend.base_loader import DatasetLoader
import pandas as pd
from backend.data_analyzer import DataAnalyzer
from backend.data_cleaner import DataCleaner
from backend.encoder import DataEncoder
from backend.data_scaler import DataScaler
from backend.feature_selector import FeatureSelector
from backend.eda import EDA
from backend.ml import ML
from backend.evaluate import Evaluation
class DataManager:
    def __init__(self, loader:DatasetLoader):
        self.loader = loader
        self.data: pd.DataFrame | None = None
        self._analyzer = None
        self._cleaner = None
        self._encoder = None
        self._scaler = None
        self._ml = None
        self._feature_selector = None
        self._eda = None
        self._evaluation = None
        
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

    @property
    def encoder(self) -> DataEncoder:
        if self._encoder is None:
            self._encoder = DataEncoder(self)
        return self._encoder

    @property
    def scaler(self) -> DataScaler:
        if self._scaler is None:
            self._scaler = DataScaler(self)
        return self._scaler

    @property
    def feature_selector(self) -> FeatureSelector:
            if self._feature_selector is None:
                self._feature_selector = FeatureSelector(self)
            return self._feature_selector

    @property
    def eda(self) -> EDA:
        if self._eda is None:
            self._eda = EDA(self)
        return self._eda
    @property
    def ml(self) -> ML:
        if self._ml is None:
            self._ml = ML(self)
        return self._ml

    @property
    def evaluation(self) -> Evaluation:
        if self._evaluation is None:
            self._evaluation = Evaluation()
        return self._evaluation