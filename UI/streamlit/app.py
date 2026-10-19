import streamlit as st
from backend.factory import LoaderFactory
from backend.data_manager import DataManager

st.title("DataFrame Only Test")

uploaded_file = st.file_uploader("Upload Dataset")

if uploaded_file is not None:

    if (
        "manager" not in st.session_state
        or uploaded_file.name != st.session_state.file_name
    ):
        loader = LoaderFactory.create_loader(uploaded_file)
        manager = DataManager(loader)
        manager.load()

        st.session_state.manager = manager
        st.session_state.file_name = uploaded_file.name

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
        target_column = st.selectbox("Select the target column", options=data.columns)
    else:
        st.warning("Dataset is Empty.")
        st.stop()

    task = st.selectbox(
        "Select the Ml task",
        options=["classification", "regression"]
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

    model_name = st.selectbox("Select the ML model", options=models[task])

    if st.button("Train Model"):
        pipeline, y_test, y_pred = manager.ml.train(
        target=target_column,
        task=task,
        model=model_name
        )
        st.write("Model Trained.")