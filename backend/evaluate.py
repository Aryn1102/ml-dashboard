from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, root_mean_squared_error, mean_absolute_error, r2_score

class Evaluation:
    def evaluate(self, y_test, y_pred, task):
        if len(y_test) != len(y_pred):
            raise ValueError("sample size doesnt match.")
        
        if task == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average = "macro")
            f1 = f1_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average="macro")
            matrix = confusion_matrix(y_test, y_pred)
            return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": matrix}
        
        elif task == "regression":
            rmse = root_mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return {"rmse": rmse, "mae": mae, "r2": r2}
        
        else:
            raise ValueError("Task is not defined. Please choose either 'classification' or 'regression'.")