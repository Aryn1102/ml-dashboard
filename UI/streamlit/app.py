import streamlit as st
from backend.factory import LoaderFactory
from backend.data_manager import DataManager

st.title("Universal Data Dashboard")

uploaded_file = st.file_uploader("Upload Dataset")

if uploaded_file is not None:
    if 'manager' not in st.session_state or uploaded_file.name != st.session_state.file_name:
        loader = LoaderFactory.create_loader(uploaded_file)
        manager = DataManager(loader)
        manager.load()
        st.session_state.manager = manager
        st.session_state.file_name = uploaded_file.name

    st.dataframe(st.session_state.manager.data)

    st.subheader("Dataset Shape")
    st.write(st.session_state.manager.analyzer.shape())
    st.subheader("Columns")
    st.write(st.session_state.manager.analyzer.columns())
    st.subheader("Missing values")
    st.write(st.session_state.manager.analyzer.missing_values())
    st.subheader("Datatypes")
    st.write(st.session_state.manager.analyzer.dtypes())
    st.subheader("Duplicates")
    st.write(st.session_state.manager.analyzer.duplicates())
    if st.button("Normalize Missing Values"):
        st.session_state.manager.cleaner.normalize_missing_values()
        st.success("Missing-value representations normalized.")
    if st.button("Remove Duplicates"):
        before = st.session_state.manager.analyzer.shape()[0]
        st.session_state.manager.cleaner.remove_duplicates()
        after = st.session_state.manager.analyzer.shape()[0]
        st.success(f"Removed {before - after} duplicate rows.")
    