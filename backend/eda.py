from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backend.data_manager import DataManager

class EDA:
    def __init__(self, manager: "DataManager") -> None:
        self.manager = manager

    def _ensure_loaded(self) -> None:
            if self.manager.data is None:
                raise ValueError("Dataset not loaded. Call DataManager.load() first.")
            
    def distributions(self, column) -> dict:
        self._ensure_loaded()
        if not column in self.manager.data.columns:
            raise ValueError("Column doesnt exist")
        if not column in self.manager.analyzer.numeric_columns():
             raise ValueError("Column is not numeric")
        
        stats = {}
        stats["mean"] = self.manager.data[column].mean()
        stats["median"] = self.manager.data[column].median()
        stats["mode"] = self.manager.data[column].mode().to_list()
        stats["min"] = self.manager.data[column].min()
        stats["max"] = self.manager.data[column].max()
        quantiles = self.manager.data[column].quantile([0.25, 0.75])
        stats["quantiles"] = {
            "Q1": quantiles.loc[0.25],
            "Q3": quantiles.loc[0.75]
        }
        stats["skew"] = self.manager.data[column].skew()
        stats["kurtosis"] = self.manager.data[column].kurt()

        return stats
        
    def outliers(self, column) -> dict:
        self._ensure_loaded()
        if not column in self.manager.data.columns:
             raise ValueError("Invalid Column")
        if not column in self.manager.analyzer.numeric_columns():
             raise ValueError("Not a numerical column.")
        
        outlier = {}
        q1, q3 = self.manager.data[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier["iqr"] = iqr
        outlier["lower_bound"] = q1 - 1.5 * iqr
        outlier["upper_bound"] = q3 + 1.5 * iqr
        mask = (
                (self.manager.data[column] < outlier["lower_bound"]) |
                (self.manager.data[column] > outlier["upper_bound"])
        )
        outlier_indices = self.manager.data.index[mask].tolist()
        outlier_data = self.manager.data.loc[mask, column].tolist()
        outlier["outlier_indices"] = outlier_indices
        outlier["outlier_data"] = outlier_data
        
        return outlier

    def correlation(self, column):
        pass