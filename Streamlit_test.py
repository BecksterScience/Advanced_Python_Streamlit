import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add the directory containing Merged_Python to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Individual Task Py Files')))

import Merged_Python as mp
import importlib
import time
import matplotlib.pyplot as plt

# Ensure module functions are accessible
def main():
    importlib.reload(mp)
    st.title("Analyzing Most Efficient Search/Sort/Filter/Group Algorithms")

    # Step 1: File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"] )
    if uploaded_file is None:
        st.info("Please upload a CSV file to get started.")
        return

    # Read the CSV into a DataFrame
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Sidebar: choose operation
    operation = st.sidebar.selectbox(
        "Select Operation", 
        ["Sorting", "Filtering", "Searching", "Grouping"]
    )

    if operation == "Sorting":
        st.sidebar.subheader("Sorting Options")
        # Choose column and algorithm
        col = st.sidebar.selectbox("Column to sort", df.columns.tolist())
        alg = st.sidebar.selectbox("Algorithm", list(mp.SORTING_ALGORITHMS.keys()))
        asc = st.sidebar.radio("Order", ["Ascending", "Descending"]) == "Ascending"

        if st.sidebar.button("Run Sorting"):
            df_copy = df.copy()
            result_df, elapsed = mp.SORTING_ALGORITHMS[alg](df_copy, col, ascending=asc)
            st.write(f"**{alg}** on *{col}* took **{elapsed:.4f}s**")
            st.subheader("Sorted Data")
            st.dataframe(result_df.head())

    elif operation == "Filtering":
        st.sidebar.subheader("Filtering Options")
        # Numeric columns only
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            st.warning("No numeric columns available for filtering.")
        else:
            col = st.sidebar.selectbox("Column to filter", num_cols)
            threshold = st.sidebar.number_input("Threshold (>)", value=float(df[col].min()))
            algs = ["filter_baseline", "filter_data_pandas", "filter_data_numpy", "filter_data_numba_wrapper"]
            alg = st.sidebar.selectbox("Filter Algorithm", algs)

            if st.sidebar.button("Run Filtering"):
                try:
                    # Use the updated filter functions that operate on DataFrame
                    filtered = getattr(mp, alg)(df.copy(), col, threshold)
                    # If NumPy array, convert to DataFrame
                    if isinstance(filtered, np.ndarray):
                        filtered = pd.DataFrame(filtered, columns=df.columns)
                    st.write(f"**{alg}** returned **{len(filtered)}** rows")
                    st.dataframe(filtered.head())
                except Exception as e:
                    st.error(f"Error in filtering: {e}")

    elif operation == "Searching":
        st.sidebar.subheader("Searching Options")
        text_cols = df.columns.tolist()
        col = st.sidebar.selectbox("Column to search", text_cols)
        value = st.sidebar.text_input("Search value")
        algs = ["search_baseline", "search_data_pandas", "search_data_numpy", "search_optimization"]
        alg = st.sidebar.selectbox("Search Algorithm", algs)

        if st.sidebar.button("Run Searching"):
            try:
                # Use the updated search functions that operate on DataFrame
                results = getattr(mp, alg)(df.copy(), col, value)
                # Convert if needed
                if isinstance(results, np.ndarray):
                    results = pd.DataFrame(results, columns=df.columns)
                st.write(f"**{alg}** found **{len(results)}** rows")
                st.dataframe(results.head())
            except Exception as e:
                st.error(f"Error in searching: {e}")

    elif operation == "Grouping":
        st.sidebar.subheader("Grouping Options")
        group_algs = [
            "group_and_aggregate_normal", "group_and_aggregate_optimized",
            "group_by_single_field_vectorized", "group_by_single_field_loop", "group_by_single_field_numba",
            "group_multiple_aggregates_vectorized", "group_multiple_aggregates_loop",
            "group_by_computed_column_vectorized", "group_by_computed_column_loop",
            "group_multiple_aggregates_vectorized"
        ]
        alg = st.sidebar.selectbox("Grouping Algorithm", group_algs)

        if alg in ["group_and_aggregate_normal"]:
            cols = st.sidebar.multiselect("Group Columns", df.columns.tolist())
            aggs = st.sidebar.text_input("Agg Dict (e.g. {'col1':'sum','col2':'mean'})")
            if st.sidebar.button("Run Grouping"):
                try:
                    agg_dict = eval(aggs)
                    result, elapsed = mp.group_and_aggregate_normal(df.copy(), cols, agg_dict)
                    st.write(f"**{alg}** took {elapsed:.4f}s")
                    st.dataframe(result.head())
                except Exception as e:
                    st.error(f"Error: {e}")
        # Additional grouping functions can be added following similar pattern

if __name__ == "__main__":
    main()
