from typing import TYPE_CHECKING

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
if TYPE_CHECKING:
    from backend.data_manager import DataManager
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
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
        categorical_features = X_train.select_dtypes(include=['object', 'category', 'string']).columns.to_list()
        numeric_features = X_train.select_dtypes(include='number').columns.to_list()
        if model == "random_forest":
            numeric_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                ]
            )
        else:
            numeric_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]
            )
        categorical_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ]
        )
        if task == "regression":
            if model == "linear_regression":
                clf = Pipeline(
                    steps=[
                        ('preprocessor', preprocessor),
                        ('regressor', LinearRegression()),
                    ]
                )
            else:
                clf = Pipeline(
                    steps=[
                        ('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(random_state=42)),
                    ]
                )
        else:
            if model == "logistic_regression":
                clf = Pipeline(
                    steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(max_iter=1000)),
                    ]
                )
            else:
                clf = Pipeline(
                    steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42)),
                    ]
                )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return clf, y_test, y_pred