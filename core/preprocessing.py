import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def split_features_target(df, target_column):
    if target_column not in df.columns:
        raise ValueError(f"{target_column} not found in DataFrame.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def build_preprocessor(x:pd.DataFrame):
    numeric_cols=x.select_dtypes(include=["int64","float64"]).columns
    categorical_cols=x.select_dtypes(include=["object", "category"]).columns
    
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cats", categorical_pipeline, categorical_cols)
        ]
    )

    return preprocessor