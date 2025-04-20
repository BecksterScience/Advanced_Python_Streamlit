import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import importlib
import time
import matplotlib.pyplot as plt

# 1) Make sure Python can find your custom modules
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "Individual Task Py Files"
        )
    )
)

# 2) Import your merged module (which contains grouping, sorting, etc.)
import Merged_Python as mp

# 3) Monkey‐patch ProcessPoolExecutor into that module’s namespace
from concurrent.futures import ProcessPoolExecutor
mp.ProcessPoolExecutor = ProcessPoolExecutor

def plot_scaling(data, func, col, *args, **kwargs):
    subset_percentages = list(range(2, 101, 2))
    total_rows = len(data)
    times = []
    subset_sizes = []

    progress_bar = st.progress(0)
    for idx, percent in enumerate(subset_percentages):
        subset_size = int((percent / 100) * total_rows)
        if subset_size == 0:
            continue

        subset_data = data.sample(n=subset_size).reset_index(drop=True)
        start = time.time()
        func(subset_data.copy(), col, *args, **kwargs)
        times.append(time.time() - start)
        subset_sizes.append(subset_size)
        progress_bar.progress((idx + 1) / len(subset_percentages))

    fig, ax = plt.subplots()
    ax.plot(subset_sizes, times, marker='o')
    ax.set_xlabel("Subset Size")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title(f"Scaling of {func.__name__} on '{col}'")
    st.pyplot(fig)

def main():
    importlib.reload(mp)

    st.set_page_config(page_title="Efficient Data Analysis App", layout="wide")
    st.title("🔍 Efficient Data Analysis App")
    st.markdown(
        "Upload a CSV and explore **Sorting**, **Filtering**, **Searching**, and **Grouping** with timings."
    )

    if 'Run' not in st.session_state:
        st.session_state['Run'] = False

    # — Upload & Preview —
    with st.expander("📂 Upload and Preview CSV", expanded=True):
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if not uploaded_file:
            st.info("Please upload a CSV file.")
            return

        try:
            df = pd.read_csv(uploaded_file)
            df.reset_index(drop=True, inplace=True)
            st.success("CSV loaded!")
            if not st.session_state['Run']:
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error: {e}")
            return

    # — Choose Operation —
    op = st.sidebar.selectbox(
        "🛠️ Select Operation",
        ["Sorting", "Filtering", "Searching", "Grouping"]
    )

    # 1) Sorting
    if op == "Sorting":
        with st.expander("⚙️ Sorting", expanded=True):
            col = st.sidebar.selectbox("Column", df.columns)
            asc = st.sidebar.radio("Order", ["Asc", "Desc"]) == "Asc"
            alg = st.sidebar.selectbox("Algorithm", list(mp.SORTING_ALGORITHMS.keys()))
            viz = st.sidebar.checkbox("Visualize Scaling")

            if st.sidebar.button("Run Sorting"):
                with st.spinner("Sorting..."):
                    d = df.copy().reset_index(drop=True)
                    sorted_df_col, t = mp.SORTING_ALGORITHMS[alg](d[[col]].copy(), col, ascending=asc)
                    d[col] = sorted_df_col[col].values
                    result = d.sort_values(col, ascending=asc).reset_index(drop=True)
                    st.session_state['Run'] = True

                st.success(f"Sorted with {alg} in {t:.4f}s")
                st.metric("Time (s)", f"{t:.4f}")
                st.dataframe(result)

                if viz:
                    plot_scaling(df.copy(), mp.SORTING_ALGORITHMS[alg], col, ascending=asc)

    # 2) Filtering
    elif op == "Filtering":
        with st.expander("🧪 Filtering", expanded=True):
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                st.warning("No numeric columns.")
            else:
                col = st.sidebar.selectbox("Column", num_cols)
                thr = st.sidebar.number_input("Threshold", float(df[col].min()), float(df[col].max()), float(df[col].min()))
                alg = st.sidebar.selectbox("Algorithm", mp.FILTER_ALGORITHMS if hasattr(mp, 'FILTER_ALGORITHMS') else ["filter_baseline"])
                viz = st.sidebar.checkbox("Visualize Scaling")

                if st.sidebar.button("Run Filtering"):
                    with st.spinner("Filtering..."):
                        res = getattr(mp, alg)(df.copy(), col, thr)
                        if isinstance(res, np.ndarray):
                            res = pd.DataFrame(res, columns=df.columns)
                        st.session_state['Run'] = True

                    st.success(f"Filtered with {alg}: {len(res)} rows in {(res.shape[0]/df.shape[0])*100:.1f}%")
                    st.dataframe(res)

                    if viz:
                        plot_scaling(df.copy(), getattr(mp, alg), col, thr)

    # 3) Searching
    elif op == "Searching":
        with st.expander("🔍 Searching", expanded=True):
            col = st.sidebar.selectbox("Column", df.columns)
            vals = sorted(df[col].dropna().unique())
            val = st.sidebar.selectbox("Value", vals)
            alg = st.sidebar.selectbox("Algorithm", ["search_baseline", "search_data_pandas"])
            viz = st.sidebar.checkbox("Visualize Scaling")

            if st.sidebar.button("Run Searching"):
                with st.spinner("Searching..."):
                    start = time.time()
                    res = getattr(mp, alg)(df.copy(), col, val)
                    t = time.time() - start
                    if isinstance(res, np.ndarray):
                        res = pd.DataFrame(res, columns=df.columns)
                    st.session_state['Run'] = True

                st.success(f"Found {len(res)} rows in {t:.4f}s with {alg}")
                st.metric("Time (s)", f"{t:.4f}")
                st.dataframe(res)

                if viz:
                    plot_scaling(df.copy(), getattr(mp, alg), col, val)

    # 4) Grouping & Aggregation
    elif op == "Grouping":
        with st.expander("📊 Grouping", expanded=True):
            group_cols = st.multiselect("Group by", df.columns)
            num_cols = df.select_dtypes("number").columns.tolist()
            agg_cols = st.multiselect("Aggregate", num_cols)
            agg_fn = st.radio("Function", ["sum", "mean", "min", "max", "count"])

            if st.button("Run Grouping Comparison"):
                if not group_cols or not agg_cols:
                    st.warning("Select at least one group and one numeric column.")
                else:
                    normal_df, t1 = mp.group_and_aggregate_normal(df, group_cols, {c:agg_fn for c in agg_cols})
                    _, t2 = mp.group_and_aggregate_optimized(df, group_cols, agg_fn, agg_cols)
                    st.subheader("Result")
                    st.dataframe(normal_df)
                    st.markdown("**Performance**")
                    st.write(f"- Normal: {t1:.4f}s")
                    st.write(f"- Optimized: {t2:.4f}s")
                    if t1 < t2:
                        st.success("Normal faster!")
                    elif t2 < t1:
                        st.success("Optimized faster!")
                    else:
                        st.info("Tie!")

if __name__ == "__main__":
    main()
