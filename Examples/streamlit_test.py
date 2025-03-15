import streamlit as st
import pandas as pd

st.title("CSV Column Viewer")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    column = st.selectbox("Select a column to view:", df.columns)
    st.write(f"### Viewing Column: {column}")
    st.write(df[[column]])