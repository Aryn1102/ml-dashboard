import streamlit as st
import pandas as pd

st.title("My ML-Dashboard")
st.write("Welcome to the app! This is where your journey begins.")
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']}
st.dataframe(pd.DataFrame(data))