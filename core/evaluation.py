from sklearn.metrics import accuracy_score, r2_score


def evaluate_model(problem_type, y_test, predictions):
    if problem_type == "classification":
        score = accuracy_score(y_test, predictions)
        return {"accuracy": score}
    else:
        score = r2_score(y_test, predictions)
        return {"r2_score": score}