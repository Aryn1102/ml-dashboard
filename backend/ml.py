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
import pandas as pd
import math
import numpy as np
class ML:
    def __init__(self, manager: "DataManager") -> None:
        self.manager = manager

    def _ensure_loaded(self):
        if self.manager.data is None:
            raise ValueError("Dataset not Loaded. Call Datamanager.load() first")
        
    def train(self, target:str, task:str, model:str) -> tuple[Pipeline, pd.Series, np.ndarray]:
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
        datetime_features = X.select_dtypes(
                                include=["datetime", "datetimetz"]
                            ).columns.to_list()
        if datetime_features:
            raise ValueError(
                f"Datetime features are not supported yet: {datetime_features}. "
                "Convert them into numerical features before training."
            )
        categorical_features = X.select_dtypes(include=['object', 'category', 'string', 'bool', 'boolean']).columns.to_list()
        numeric_features = X.select_dtypes(include='number').columns.to_list()
        supported_features = set(numeric_features + categorical_features)
        unsupported_features = [
            column for column in X.columns
            if column not in supported_features
        ]
        if unsupported_features:
            raise ValueError(
                f"Unsupported feature types: {unsupported_features}. "
                "Convert these columns before training."
            )
        if X.shape[1]==0:
            raise ValueError(
                "No feature columns present. "
                "The dataset must contain at least one feature besides the target."
            )
        y = self.manager.data[target]
        if y.isna().any():
            raise ValueError(
                "Target column contains missing values. "
                "Remove or resolve them before training."
            )
        if task == "regression" and not pd.api.types.is_numeric_dtype(y):
            raise ValueError(
                "Regression requires a numeric target column."
            )
        if task == "classification" and (y.nunique() <2):
            raise ValueError(
                "Classification requires atleast 2 classes."
            )
        if len(X) < 5:
            raise ValueError("No. of rows of feature columns are less than 5.")
        stratify = None
        if task == "classification":
            stratify = y
            class_counts = y.value_counts()
            n_classes = len(class_counts)
            n_samples = len(y)
            test_size = 0.2
            n_test = int(math.ceil(n_samples * test_size))
            n_train = n_samples - n_test
            if class_counts.min() < 2:
                raise ValueError(
                    "Each class must contain at least 2 samples "
                    "for stratified train/test splitting."
                )

            if n_test < n_classes:
                raise ValueError(
                    "Test set is too small to represent every class. "
                    "Increase the dataset size or test-set proportion."
                )

            if n_train < n_classes:
                raise ValueError(
                    "Training set is too small to represent every class. "
                    "Increase the dataset size or test-set proportion."
                )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
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