from core.preprocessing import split_features_target, build_preprocessor
from core.model import detect_problem_type, get_model, build_pipeline
from core.evaluation import evaluate_model
from sklearn.model_selection import train_test_split
from core.target_resolver import resolve_target
from sklearn.model_selection import cross_val_score
import joblib

def train_model(df, target_column=None):
    if df.empty:
        raise ValueError("Dataset is empty after loading.")
    
    target_column = resolve_target(df, target_column)
    X, y = split_features_target(df, target_column)

    preprocessor = build_preprocessor(X)

    problem_type = detect_problem_type(y)

    model = get_model(problem_type)

    pipeline = build_pipeline(preprocessor, model)

    scores = cross_val_score(pipeline, X, y, cv=5)

    mean_score = scores.mean()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    metrics = evaluate_model(problem_type, y_test, predictions)

    return {
    "pipeline": pipeline,
    "test_metrics": metrics,
    "cv_scores": scores.tolist(),
    "cv_mean_score": mean_score
    }

def get_candidate_models(problem_type):

    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

    if problem_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(),
            "gradient_boosting": GradientBoostingClassifier()
        }

    else:
        return {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(),
            "gradient_boosting": GradientBoostingRegressor()
        }