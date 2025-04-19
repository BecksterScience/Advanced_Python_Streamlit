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

def plot_scaling(data, func, col, *args, **kwargs):
    subset_percentages = [i for i in range(2, 101, 2)]
    total_rows = len(data)
    times = []
    subset_sizes = []

    progress_bar = st.progress(0)
    for idx, percent in enumerate(subset_percentages):
        subset_size = int((percent / 100) * total_rows)
        if subset_size == 0:
            continue

        subset_data = data.sample(n=subset_size)
        start_time = time.time()
        func(subset_data.copy(), col, *args, **kwargs)
        times.append(time.time() - start_time)
        subset_sizes.append(subset_size)
        progress_bar.progress((idx + 1) / len(subset_percentages))

    fig, ax = plt.subplots()
    ax.plot(subset_sizes, times, marker='o')
    ax.set_xlabel("Subset Size")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title(f"Scaling Performance of {func.__name__} on '{col}'")
    st.pyplot(fig)

def main():
    importlib.reload(mp)
    st.set_page_config(page_title="Efficient Data Analysis App", layout="wide")
    st.title("🔍 Efficient Data Analysis App")
    st.markdown("Upload a CSV file and explore optimized **Sorting**, **Filtering**, **Searching**, and **Grouping** algorithms with performance metrics.")

    if 'Run' not in st.session_state:
        st.session_state['Run'] = False

    with st.expander("📂 Upload and Preview CSV"):
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is None:
            st.info("Please upload a CSV file to get started.")
            return

        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV loaded successfully!")
            if not st.session_state['Run']:
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

    operation = st.sidebar.selectbox("🛠️ Select Operation", ["Sorting", "Filtering", "Searching", "Grouping"])

    if operation == "Sorting":
        with st.expander("⚙️ Sorting Options", expanded=True):
            col = st.sidebar.selectbox("Column to sort", df.columns.tolist())
            if pd.api.types.is_numeric_dtype(df[col]):
                st.sidebar.markdown(f"**Stats for `{col}`**")
                st.sidebar.metric("Min", f"{df[col].min():.2f}")
                st.sidebar.metric("Max", f"{df[col].max():.2f}")
                st.sidebar.metric("Mean", f"{df[col].mean():.2f}")
                st.sidebar.metric("Std Dev", f"{df[col].std():.2f}")
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                top_vals = df[col].value_counts().head(3)
                st.sidebar.markdown(f"**Top Values in `{col}`**")
                for val, count in top_vals.items():
                    st.sidebar.markdown(f"`{val}`: {count} rows")

            alg = st.sidebar.selectbox("Sorting Algorithm", list(mp.SORTING_ALGORITHMS.keys()))
            asc = st.sidebar.radio("Sort Order", ["Ascending", "Descending"]) == "Ascending"
            visualize = st.sidebar.checkbox("📈 Visualize Scaling Performance")

            if st.sidebar.button("Run Sorting"):
                with st.spinner("Sorting..."):
                    df_copy = df.copy()
                    sorted_col_df, elapsed = mp.SORTING_ALGORITHMS[alg](df_copy[[col]].copy(), col, ascending=asc)
                    sorted_values = sorted_col_df[col].values
                    temp_sorted_df = df_copy.copy()
                    temp_sorted_df[col] = sorted_values
                    result_df = temp_sorted_df.sort_values(by=col, ascending=asc).reset_index(drop=True)
                    st.session_state['Run'] = True

                st.success(f"Sorted using **{alg}** in **{elapsed:.4f}s**")
                st.metric("Execution Time (s)", f"{elapsed:.4f}")
                st.subheader("🔢 Resulting Data Preview")
                st.dataframe(result_df)

                if visualize:
                    st.subheader("📊 Sorting Performance Scaling")
                    plot_scaling(df.copy(), mp.SORTING_ALGORITHMS[alg], col, ascending=asc)

    elif operation == "Filtering":
        with st.expander("🧪 Filtering Options", expanded=True):
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                st.warning("No numeric columns available for filtering.")
            else:
                col = st.sidebar.selectbox("Column to filter", num_cols)
                col_min, col_max = df[col].min(), df[col].max()
                st.sidebar.markdown(f"**Stats for `{col}`**")
                st.sidebar.metric("Min", f"{col_min:.2f}")
                st.sidebar.metric("Max", f"{col_max:.2f}")
                st.sidebar.metric("Mean", f"{df[col].mean():.2f}")
                st.sidebar.metric("Std Dev", f"{df[col].std():.2f}")
                threshold = st.sidebar.number_input("Threshold (>)", min_value=float(col_min), max_value=float(col_max), value=float(col_min))
                alg = st.sidebar.selectbox("Filter Algorithm", ["filter_baseline", "filter_data_pandas", "filter_data_numpy", "filter_data_numba_wrapper"])
                visualize = st.sidebar.checkbox("📈 Visualize Scaling Performance")

                if st.sidebar.button("Run Filtering"):
                    with st.spinner("Filtering..."):
                        result_df = getattr(mp, alg)(df.copy(), col, threshold)
                        if isinstance(result_df, np.ndarray):
                            result_df = pd.DataFrame(result_df, columns=df.columns)
                        st.session_state['Run'] = True
                    st.success(f"Filtered with **{alg}** — **{len(result_df)} rows** returned")
                    st.subheader("📄 Resulting Data Preview")
                    st.dataframe(result_df)

                    if visualize:
                        st.subheader("📊 Filtering Performance Scaling")
                        plot_scaling(df.copy(), getattr(mp, alg), col, threshold)

    elif operation == "Searching":
        with st.expander("🔍 Searching Options", expanded=True):
            col = st.sidebar.selectbox("Column to search", df.columns.tolist())
            unique_vals = sorted(df[col].dropna().unique())
            value = st.sidebar.selectbox("Search value", unique_vals, index=0)
            alg = st.sidebar.selectbox("Search Algorithm", ["search_baseline", "search_data_pandas", "search_data_numpy", "search_optimization"])
            visualize = st.sidebar.checkbox("📈 Visualize Scaling Performance")

            if st.sidebar.button("Run Searching"):
                with st.spinner("Searching..."):
                    df_copy = df.copy()
                    start_time = time.time()
                    result_df = getattr(mp, alg)(df_copy, col, value)
                    elapsed = time.time() - start_time
                    if isinstance(result_df, np.ndarray):
                        result_df = pd.DataFrame(result_df, columns=df.columns)
                    st.session_state['Run'] = True

                st.success(f"Search using **{alg}** found **{len(result_df)} rows** in **{elapsed:.4f}s**")
                st.metric("Execution Time (s)", f"{elapsed:.4f}")
                st.subheader("🔍 Resulting Data Preview")
                st.dataframe(result_df)

                if visualize:
                    st.subheader("📊 Searching Performance Scaling")
                    plot_scaling(df.copy(), getattr(mp, alg), col, value)

    elif operation == "Grouping":
        with st.expander("📊 Grouping Options", expanded=True):
            group_algs = [
                "group_and_aggregate_normal", "group_and_aggregate_optimized",
                "group_by_single_field_vectorized", "group_by_single_field_loop", "group_by_single_field_numba",
                "group_multiple_aggregates_vectorized", "group_multiple_aggregates_loop",
                "group_by_computed_column_vectorized", "group_by_computed_column_loop"
            ]
            alg = st.sidebar.selectbox("Grouping Algorithm", group_algs)
            visualize = st.sidebar.checkbox("📈 Visualize Scaling Performance")

            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            non_numeric_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            all_cols = df.columns.tolist()

            if alg == "group_and_aggregate_normal":
                group_cols = st.sidebar.multiselect("Group by column(s)", all_cols)
                selected_agg_cols = st.sidebar.multiselect("Columns to aggregate", numeric_cols)
                agg_func = st.sidebar.selectbox("Aggregation Function", ["mean", "sum", "min", "max", "count"])

                if st.sidebar.button("Run Grouping"):
                    if not group_cols or not selected_agg_cols:
                        st.warning("Please select both group columns and numeric aggregation columns.")
                    else:
                        with st.spinner("Grouping..."):
                            agg_dict = {col: agg_func for col in selected_agg_cols}
                            result_df, elapsed = mp.group_and_aggregate_normal(df.copy(), group_cols, agg_dict)
                            st.session_state['Run'] = True
                        st.success(f"Grouped in **{elapsed:.4f}s**")
                        st.subheader("📊 Resulting Data Preview")
                        st.dataframe(result_df)

            elif alg == "group_and_aggregate_optimized":
                group_cols = st.sidebar.multiselect("Group by column(s)", all_cols)
                selected_agg_cols = st.sidebar.multiselect("Columns to aggregate", numeric_cols)
                agg_func = st.sidebar.selectbox("Aggregation Function", ["mean", "sum", "min", "max", "count"])

                if st.sidebar.button("Run Grouping"):
                    if not group_cols or not selected_agg_cols:
                        st.warning("Please select both group columns and numeric aggregation columns.")
                    else:
                        with st.spinner("Grouping..."):
                            result_df, elapsed = mp.group_and_aggregate_optimized(df.copy(), group_cols, agg_func, selected_agg_cols)
                            st.session_state['Run'] = True
                        st.success(f"Grouped in **{elapsed:.4f}s**")
                        st.subheader("📊 Resulting Data Preview")
                        st.dataframe(result_df)

            elif alg.startswith("group_by_single_field"):
                col = st.sidebar.selectbox("Column to Group By", all_cols)
                st.sidebar.metric("Unique Values", df[col].nunique())
                top_vals = df[col].value_counts().head(3)
                st.sidebar.markdown("**Top Categories**")
                for val, count in top_vals.items():
                    st.sidebar.markdown(f"`{val}`: {count} rows")

                if st.sidebar.button("Run Grouping"):
                    result_df, elapsed = getattr(mp, alg)(df.copy(), col)
                    st.session_state['Run'] = True
                    st.success(f"Grouped in **{elapsed:.4f}s**")
                    st.subheader("📊 Resulting Data Preview")
                    st.dataframe(result_df)

            elif alg.startswith("group_multiple_aggregates"):
                group_col = st.sidebar.selectbox("Group Column", all_cols)
                count_col = st.sidebar.selectbox("Column to Count", all_cols)
                numeric_col = st.sidebar.selectbox("Column to Average", numeric_cols)
                if st.sidebar.button("Run Grouping"):
                    result_df, elapsed = getattr(mp, alg)(df.copy(), group_col, count_col, numeric_col)
                    st.session_state['Run'] = True
                    st.success(f"Grouped in **{elapsed:.4f}s**")
                    st.subheader("📊 Resulting Data Preview")
                    st.dataframe(result_df)

            elif alg.startswith("group_by_computed_column"):
                col = st.sidebar.selectbox("Column to bucket/group by computed value", all_cols)
                numeric_col = st.sidebar.selectbox("Numeric Column to Average", numeric_cols)
                st.sidebar.markdown("This will group by a computed transformation of the selected column (e.g., bucketed values).")
                if st.sidebar.button("Run Grouping"):
                    result_df, elapsed = getattr(mp, alg)(df.copy(), col, numeric_col)
                    if result_df is not None:
                        st.session_state['Run'] = True
                        st.success(f"Grouped in **{elapsed:.4f}s**")
                        st.subheader("📊 Resulting Data Preview")
                        st.dataframe(result_df)
                    else:
                        st.error("Invalid column or grouping transformation failed.")



if __name__ == "__main__":
    main()
