import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from backend.factory import LoaderFactory
from backend.data_manager import DataManager
from backend.evaluate import Evaluation
import pandas as pd
import hashlib

st.set_page_config(
    page_title="Universal ML Dashboard",
    page_icon="🧠",
    layout="wide",
)

def clear_metrics():
    st.session_state.pop("metrics", None)
    st.session_state.pop("trained_task", None)
    st.session_state.pop("pipeline", None)
    st.session_state.pop("training_complete", None)

st.title("Universal ML Dashboard")
st.caption("Explore datasets, prepare data, and train machine learning models.")

uploaded_file = st.file_uploader("Upload Dataset")
def show_evaluation(metrics, trained_task):
    with st.container(border=True):
        st.subheader("Model Evaluation")
        st.caption(f"Task: {trained_task.title()}")
        if trained_task == "classification":
            accuracy, precision, f1, recall = st.columns(4)

            with accuracy:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                st.caption("Proportion of predictions that were correct.")

            with precision:
                st.metric("Precision", f"{metrics['precision']:.3f}")
                st.caption("Macro-average precision across classes.")

            with f1:
                st.metric("F1 Score", f"{metrics['f1']:.3f}")
                st.caption("Harmonic mean of precision and recall.")

            with recall:
                st.metric("Recall", f"{metrics['recall']:.3f}")
                st.caption("Macro-average recall across classes.")

            matrix = pd.DataFrame(
                metrics["confusion_matrix"],
                index=metrics["labels"],
                columns=metrics["labels"],
            )

            st.subheader("Confusion Matrix")
            st.caption(
                "Rows represent actual classes; columns represent predicted classes."
            )
            st.dataframe(matrix, use_container_width=True)

        elif trained_task == "regression":
            rmse, mae, r2 = st.columns(3)

            with rmse:
                st.metric("RMSE", f"{metrics['rmse']:.3f}")
                st.caption("Root mean squared error, in target units.")

            with mae:
                st.metric("MAE", f"{metrics['mae']:.3f}")
                st.caption("Mean absolute error, in target units.")

            with r2:
                st.metric("R²", f"{metrics['r2']:.3f}")
                st.caption("Coefficient of determination.")
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    filehash = hashlib.sha256(file_bytes).hexdigest()
    previous_hash = st.session_state.get("file_hash")
    if (
        "manager" not in st.session_state
        or filehash != previous_hash
    ):
        clear_metrics()
        loader = LoaderFactory.create_loader(uploaded_file)
        manager = DataManager(loader)
        manager.load()

        st.session_state.manager = manager
        st.session_state.file_name = uploaded_file.name
        st.session_state.file_hash = filehash

    manager = st.session_state.manager
    data = manager.data
    row, col = manager.analyzer.shape()
    missing_count = int(data.isna().sum().sum())
    duplicate_count = manager.analyzer.duplicates().shape[0]
    data_tab, training_tab, evaluation_tab = st.tabs(
        ["Data Overview", "Train Model", "Evaluation"]
    )

    with data_tab:
        preview_col, overview_col = st.columns([2, 1])
        with preview_col:
            with st.container(border=True):
                st.subheader("Dataset Preview")
                st.caption(
                    f"{st.session_state.file_name} · "
                    f"{row:,} rows · {col:,} columns"
                )
                st.dataframe(
                    data,
                    use_container_width=True,
                    height=350,
                )

        with overview_col:
            with st.container(border=True):
                st.subheader("Dataset Overview")

                st.metric("Rows", f"{row:,}")
                st.metric("Features", f"{col:,}")
                st.metric("Missing Values", f"{missing_count:,}")
                st.metric("Duplicate Rows", f"{duplicate_count:,}")

    with training_tab:
        with st.container(border=True):
            st.subheader("Model Training")
            st.caption("Choose a target, task, and model to begin.")

            if not data.columns.empty:
                target_column = st.selectbox(
                    "Select the target column",
                    options=data.columns,
                    key="target_column",
                    on_change=clear_metrics,
                )
            else:
                st.warning("Dataset is empty.")
                st.stop()

            task = st.selectbox(
                "Select the ML task",
                options=["classification", "regression"],
                key="task",
                on_change=clear_metrics,
            )

            models = {
                "classification": [
                    "logistic_regression",
                    "random_forest",
                ],
                "regression": [
                    "linear_regression",
                    "random_forest",
                ],
            }

            model_name = st.selectbox(
                "Select the ML model",
                options=models[task],
                key="model_name",
                on_change=clear_metrics,
            )

            train_clicked = st.button("Train Model", type="primary")
        if train_clicked:
            clear_metrics()

            try:
                with st.spinner("Training model..."):
                    pipeline, y_test, y_pred = manager.ml.train(
                        target=target_column,
                        task=task,
                        model=model_name,
                    )

                    metrics = Evaluation().evaluate(y_test, y_pred, task)

                st.session_state["metrics"] = metrics
                st.session_state["pipeline"] = pipeline
                st.session_state["trained_task"] = task
                st.session_state["training_complete"] = True

                st.success("Model trained successfully.")

            except Exception as error:
                st.session_state["training_complete"] = False
                st.error("An unexpected error occurred during training.")
                st.exception(error)

        if (
            st.session_state.get("training_complete")
            and "metrics" in st.session_state
        ):
            show_evaluation(
                st.session_state["metrics"],
                st.session_state["trained_task"],
            )

    with evaluation_tab:
        if (
            "metrics" in st.session_state
            and "trained_task" in st.session_state
            and "pipeline" in st.session_state
        ):
            show_evaluation(
                st.session_state["metrics"],
                st.session_state["trained_task"],
            )