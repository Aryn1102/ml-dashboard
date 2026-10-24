from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils.multiclass import unique_labels

class Evaluation:
    def evaluate(self, y_test, y_pred, task):
        if task not in ("classification", "regression"):
            raise ValueError("Task is not defined. Please choose either 'classification' or 'regression'.")
        if len(y_test) != len(y_pred):
            raise ValueError("sample size doesnt match.")
        if len(y_test) == 0:
            raise ValueError("Cannot evaluate predictions on an empty test set.")
        if task == "classification":
            labels = unique_labels(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average = "macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
            matrix = confusion_matrix(y_test, y_pred, labels=labels)
            return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": matrix, "labels": labels}
        
        else:
            rmse = root_mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return {"rmse": rmse, "mae": mae, "r2": r2}