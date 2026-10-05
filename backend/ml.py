from typing import TYPE_CHECKING

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
if TYPE_CHECKING:
    from backend.data_manager import DataManager
from sklearn.ensemble import RandomForestClassifier

class ML:
    def __init__(self, manager: "DataManager") -> None:
        self.manager = manager

    def _ensure_loaded(self):
        if self.manager.data is None:
            raise ValueError("Dataset not Loaded. Call Datamanager.load() first")
        
    def train(self, target, task, model) -> tuple:
        self._ensure_loaded()
        allowed_tasks = {"classification", "regression"}
        models = {
            "classification": {
                "logistic_regression",
                "random_forest"
            },
            "regression": {
                "linear_regression",
                "random_forest"
            }}
        if target not in self.manager.data.columns:
            raise ValueError("Target column not in dataframe")
        if task not in allowed_tasks:
            raise ValueError("Task is not defined.")
        if model not in models[task]:
            raise ValueError("Model not present")

        X = self.manager.data.drop(columns=[target])
        y = self.manager.data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        if task == "classification":
            if model == "random_forest":
                clf = RandomForestClassifier()
                clf_model = clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)
                return clf_model, y_test, y_pred