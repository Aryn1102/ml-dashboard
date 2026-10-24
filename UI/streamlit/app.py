import streamlit as st
from backend.factory import LoaderFactory
from backend.data_manager import DataManager
from backend.evaluate import Evaluation
import pandas as pd
import hashlib
def clear_metrics():
    st.session_state.pop("metrics", None)
    st.session_state.pop("trained_task", None)
    st.session_state.pop("pipeline", None)

st.title("DataFrame Only Test")

uploaded_file = st.file_uploader("Upload Dataset")

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

    st.subheader("Dataset Preview")
    st.dataframe(data)

    st.subheader("Dataset Shape")
    row, col = manager.analyzer.shape()
    st.metric("Rows", row)
    st.metric("Columns", col)

    st.subheader("Columns")
    columns = manager.analyzer.columns()
    st.dataframe(
        columns.to_frame(name="Column"),
        hide_index=True
    )

    st.subheader("Missing Values")

    missing_values = manager.analyzer.missing_values().to_frame(
        name="Missing Values"
    ).reset_index()

    missing_values = missing_values.rename(
        columns={"index": "Column"}
    )

    st.dataframe(
        missing_values,
        hide_index=True
    )
    
    st.subheader("Datatypes")
    dtypes = manager.analyzer.dtypes().astype(str).to_frame(name="Dtype").reset_index()
    dtypes = dtypes.rename(columns={"index": "Column"})
    st.dataframe(
        dtypes,
        hide_index=True
    )
    st.subheader("Duplicate")
    duplicates_shape = manager.analyzer.duplicates().shape
    st.metric("Duplicate Rows", duplicates_shape[0])

    st.subheader("Model Training")
    if not data.columns.empty:
        target_column = st.selectbox("Select the target column",
                                    options=data.columns,
                                    key="target_column",
                                    on_change=clear_metrics)
    else:
        st.warning("Dataset is Empty.")
        st.stop()

    task = st.selectbox(
        "Select the Ml task",
        options=["classification", "regression"],
        key= "task",
        on_change=clear_metrics
        )

    models={
        "classification": [
            "logistic_regression",
            "random_forest"
        ],
        "regression": [
            "linear_regression",
            "random_forest"
        ]
    }

    model_name = st.selectbox("Select the ML model", options=models[task], key="model_name", on_change=clear_metrics)

    if st.button("Train Model"):
        clear_metrics()
        try:
            pipeline, y_test, y_pred = manager.ml.train(
            target=target_column,
            task=task,
            model=model_name
            )
            metrics = Evaluation().evaluate(y_test, y_pred, task)
            st.session_state["metrics"] = metrics
            st.session_state["pipeline"] = pipeline
            st.session_state["trained_task"] = task
        except ValueError as error:
            st.error(f"Could not train model: {error}")
    if "metrics" in st.session_state and "trained_task" in st.session_state  and "pipeline" in st.session_state:
        metrics = st.session_state["metrics"]
        pipeline = st.session_state["pipeline"]
        trained_task = st.session_state["trained_task"]
        if trained_task == "classification":
            accuracy, precision, f1, recall = st.columns(4)
            with accuracy:
                accuracy.metric("Accuracy", metrics["accuracy"])
                st.caption("Accuracy: Proportion of predictions that were correct.")
            with precision:
                precision.metric("Precision", metrics["precision"])
                st.caption("Precision: Average precision across classes, giving each class equal weight.")
            with f1:
                f1.metric("F1", metrics["f1"])
                st.caption("F1: Harmonic mean of precision and recall.")
            with recall:
                recall.metric("Recall", metrics["recall"])
                st.caption("Recall: How many actual instances of a class the model identified.")
            matrix = pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
            st.subheader("Confusion Matrix")
            st.caption("Rows represent actual classes.  \nColumns represent predicted classes.  \nThe diagonal contains correct predictions; off-diagonal cells contain misclassifications.")
            st.dataframe(matrix)
        elif trained_task == "regression":
            rmse, mae, r2 = st.columns(3)
            with rmse:
                rmse.metric("RMSE", metrics["rmse"])
                st.caption("RMSE: Typical prediction error, with larger errors penalized more heavily. It uses the same units as the target.")
            with mae:
                mae.metric("MAE", metrics["mae"])
                st.caption("MAE: Average absolute difference between actual and predicted values, in the target's units.")
            with r2:
                r2.metric("R²", metrics["r2"])
                st.caption("R2: How well the model explains variation in the target relative to a mean-prediction baseline. It can be negative when the model performs worse than that baseline.")