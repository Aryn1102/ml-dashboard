from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline


def detect_problem_type(y):
    if y.dtype == "object":
        return "classification"

    unique_values = y.unique()

    if len(unique_values) == 2:
        return "classification"

    return "regression"


def get_model(problem_type):
    if problem_type == "classification":
        return LogisticRegression(max_iter=1000)
    else:
        return LinearRegression()


def build_pipeline(preprocessor, model):
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipeline