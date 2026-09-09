import streamlit as st
from backend.factory import LoaderFactory
from backend.data_manager import DataManager

st.title("Universal Data Dashboard")

uploaded_file = st.file_uploader("Upload Dataset")

if uploaded_file is not None:
    if 'manager' not in st.session_state:
        loader = LoaderFactory.create_loader(uploaded_file)
        manager = DataManager(loader)
        st.session_state.manager.load()
        st.session_state.manager = manager
        
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
    if st.button("Replace NaN"):
        st.session_state.manager.cleaner.replace_nan()