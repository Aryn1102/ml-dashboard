from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backend.data_manager import DataManager
import pandas as pd
import math
import numpy as np
class EDA:
    def __init__(self, manager: "DataManager") -> None:
        self.manager = manager

    def _ensure_loaded(self) -> None:
            if self.manager.data is None:
                raise ValueError("Dataset not loaded. Call DataManager.load() first.")
            
    def distributions(self, column) -> dict:
        self._ensure_loaded()
        if column not in self.manager.data.columns:
            raise ValueError("Column doesnt exist")
        if column not in self.manager.analyzer.numeric_columns():
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
        if column not in self.manager.data.columns:
             raise ValueError("Invalid Column")
        if column not in self.manager.analyzer.numeric_columns():
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

    def correlation(self) -> pd.DataFrame:
        self._ensure_loaded()
        num_col = self.manager.analyzer.numeric_columns()
        if len(num_col) < 2:
            raise ValueError("At least two numeric columns are required for correlation analysis.")
        corr_matrix = self.manager.data[num_col].corr(method="pearson")

        return corr_matrix

    def correlation_pairs(
        self, threshold: float
    ) -> list[tuple[str, str, float]]:
        self._ensure_loaded()

        if isinstance(threshold, bool) or not isinstance(
            threshold, (int, float, np.integer, np.floating)
        ):
            raise TypeError("Threshold must be a numeric value.")

        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be finite and between 0 and 1.")

        num_col = self.manager.analyzer.numeric_columns()
        if len(num_col) < 2:
            raise ValueError(
                "At least two numeric columns are required for correlation analysis."
            )

        corr_matrix = self.manager.data[num_col].corr(method="pearson")
        high_corr_pairs = []

        for i in range(len(num_col)):
            for j in range(i + 1, len(num_col)):
                value = corr_matrix.iloc[i, j]

                if math.isfinite(value) and abs(value) >= threshold:
                    high_corr_pairs.append((
                        corr_matrix.index[i],
                        corr_matrix.columns[j],
                        float(value)
                    ))

        return high_corr_pairs