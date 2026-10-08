from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class DataEncoder:
    def __init__(self):
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self._is_fitted = False

    def fit(self, X: pd.DataFrame) -> None:
        self.encoder.fit(X)
        self._is_fitted = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._is_fitted:
            index = X.index
            transformed_data = self.encoder.transform(X)
            column_names = self.encoder.get_feature_names_out()
            return pd.DataFrame(transformed_data, columns=column_names, index=index)
        else:
            raise ValueError("Encoder has not been fitted.")

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)