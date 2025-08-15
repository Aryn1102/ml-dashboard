import streamlit as st
import pandas as pd

st.title("My ML-Dashboard")
st.write("Welcome to the app! This is where your journey begins.")
upload=st.file_uploader("Upload the file",type=['csv'], accept_multiple_files=False)
data=pd.read_csv(upload)
st.dataframe(pd.DataFrame(data))