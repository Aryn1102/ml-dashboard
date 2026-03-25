import pandas as pd
import os


def detect_target(df):
    for col in df.columns:
        if col.lower() in ["target", "label", "class", "y"]:
            return col
    return None


def detect_problem_type(df, target):
    if target is None:
        return "Unknown"

    if df[target].nunique() <= 10:
        return "Classification"
    else:
        return "Regression"


def generate_report(df, target, problem_type, file_name):
    report = "\n=== AI Dataset Review Agent Report ===\n\n"
    report += f"Analyzed File: {file_name}\n"
    report += "Execution Context: GitLab Pipeline Trigger\n\n"

    # Overview
    report += "Overview:\n"
    report += f"- Rows: {df.shape[0]}\n"
    report += f"- Columns: {df.shape[1]}\n\n"

    # Data Quality
    report += "Data Quality:\n"
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]

    if missing_cols.empty:
        report += "- No missing values detected\n"
    else:
        for col, val in missing_cols.items():
            report += f"- {col}: {val} missing values\n"

    report += "\n"

    # ML Insights
    report += "ML Insights:\n"
    report += f"- Problem Type: {problem_type}\n"
    report += f"- Target Column: {target}\n\n"

    # Key Observations
    report += "Key Observations:\n"

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        strong_found = False

        for col1 in corr.columns:
            for col2 in corr.columns:
                if col1 != col2 and corr.loc[col1, col2] > 0.7:
                    report += f"- Strong correlation between {col1} and {col2}\n"
                    strong_found = True
                    break
            if strong_found:
                break

        if not strong_found:
            report += "- No strong correlations detected\n"

    report += "\n"

    # Recommendations
    report += "Recommendations:\n"
    report += "- Handle missing values before training\n"
    report += "- Normalize numeric features if needed\n"

    if problem_type == "Classification":
        report += "- Suitable for classification models (Random Forest, Logistic Regression)\n"
    elif problem_type == "Regression":
        report += "- Suitable for regression models (Linear Regression, XGBoost)\n"
    
    report += "\nFinal Assessment:\n"

    if problem_type != "Unknown":
        report += f"- Dataset is suitable for {problem_type} tasks\n"
    else:
        report += "- Unable to determine ML task confidently\n"

    report += "- Ready for preprocessing and model development\n"
    
    return report


def analyze_dataset(file_path):
    df = pd.read_csv(file_path)

    target = detect_target(df)
    problem_type = detect_problem_type(df, target)

    return generate_report(df, target, problem_type, file_path)


def find_csv_file():
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".csv"):
                return os.path.join(root, file)
    return None


if __name__ == "__main__":
    csv_file = find_csv_file()

    if csv_file:
        report = analyze_dataset(csv_file)

        with open("analysis_report.txt", "w") as f:
            f.write(report)

        print("Analysis complete. Report generated.")
    else:
        print("No CSV file found.")