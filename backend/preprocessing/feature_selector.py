from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backend.data_manager import DataManager

class FeatureSelector:

    def __init__(self, manager: "DataManager") -> None:
        self.manager = manager

    def _ensure_loaded(self) -> None:
        if self.manager.data is None:
            raise ValueError("Dataset not loaded. Call DataManger.load() first")

    def correlation_pairs(self,threshold)-> list:
        self._ensure_loaded()
        num_col = self.manager.analyzer.numeric_columns()
        if len(num_col) < 2:
            raise ValueError("Numeric Column are too low to create a correalation.")
        corr_matrix = self.manager.data[num_col].corr(method="pearson")
        n = corr_matrix.shape[0]
        high_corr_pairs = []
        for i in range(n):
            for j in range(i+1,n):
                value = corr_matrix.iloc[i,j]
                if abs(value) >= threshold:
                    var1 = corr_matrix.index[i]
                    var2 = corr_matrix.index[j]
                    high_corr_pairs.append((var1, var2, value))
        return high_corr_pairs