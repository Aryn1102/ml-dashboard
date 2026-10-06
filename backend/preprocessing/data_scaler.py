from sklearn.preprocessing import StandardScaler

class DataScaler:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, X) -> None:
        self.scaler.fit(X)
        self._is_fitted = True

    def transform(self, X):
        if self._is_fitted:
            return self.scaler.transform(X)
        else:
            raise ValueError("Scaler has not been fitted.")

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)