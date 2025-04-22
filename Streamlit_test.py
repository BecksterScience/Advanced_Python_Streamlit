import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import importlib
import time
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mproc
from functools import partial

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

# 3) Monkey‐patch ProcessPoolExecutor into that module's namespace
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

def reduce_memory_usage(df):
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if pd.api.types.is_numeric_dtype(col_type):
            if df[col].isnull().any():
                continue

            try:
                c_min = df[col].min()
                c_max = df[col].max()
            except:
                continue  # skipping if min/max throws error

            if pd.api.types.is_integer_dtype(col_type):
                try:
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                except:
                    continue
            else:
                try:
                    if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
                except:
                    continue

        elif pd.api.types.is_object_dtype(col_type):
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                if df[col].nunique() / len(df[col]) < 0.5:
                    df[col] = df[col].astype('category')

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    return df, start_mem, end_mem

def parallel_min(data_chunk):
    return data_chunk.min()

def parallel_max(data_chunk):
    return data_chunk.max()

def parallel_sum(data_chunk):
    return data_chunk.sum()

def parallel_count(data_chunk):
    return len(data_chunk)

def parallel_search(data_chunk, col, value):
    return data_chunk[data_chunk[col] == value]

def parallel_filter(data_chunk, col, threshold):
    return data_chunk[data_chunk[col] > threshold]

def parallel_sort(data_chunk, col, ascending=True):
    return data_chunk.sort_values(by=col, ascending=ascending)

def parallel_mean(data_chunk):
    return data_chunk.mean()

def parallel_std(data_chunk):
    return data_chunk.std()

def parallel_var(data_chunk):
    return data_chunk.var()

def parallel_correlation(data_chunk, col1, col2):
    return data_chunk[[col1, col2]].corr().iloc[0, 1]

def parallel_unique(data_chunk, col):
    return data_chunk[col].unique()

def parallel_groupby(data_chunk, group_col, agg_col, agg_func):
    if agg_func == 'sum':
        return data_chunk.groupby(group_col)[agg_col].sum()
    elif agg_func == 'mean':
        return data_chunk.groupby(group_col)[agg_col].mean()
    elif agg_func == 'count':
        return data_chunk.groupby(group_col)[agg_col].count()
    elif agg_func == 'min':
        return data_chunk.groupby(group_col)[agg_col].min()
    elif agg_func == 'max':
        return data_chunk.groupby(group_col)[agg_col].max()

# Additional helper functions to replace lambdas
def sum_column(data_chunk, col):
    return data_chunk[col].sum()

def count_rows(data_chunk):
    return len(data_chunk)

def sum_squared_diff(data_chunk, col, mean):
    return ((data_chunk[col] - mean) ** 2).sum()

def calculate_covariance(data_chunk, col1, col2, mean1, mean2):
    return ((data_chunk[col1] - mean1) * (data_chunk[col2] - mean2)).sum()

def execute_parallel_operation(df, operation, col=None, col2=None, value=None, threshold=None, num_cores=None, ascending=True, group_col=None, agg_col=None, agg_func=None):
    # For small datasets, use pandas directly
    if len(df) < 10000:  # Threshold for parallel processing
        if operation == 'min':
            return df[col].min()
        elif operation == 'max':
            return df[col].max()
        elif operation == 'sum':
            return df[col].sum()
        elif operation == 'count':
            return len(df)
        elif operation == 'search':
            return df[df[col] == value]
        elif operation == 'filter':
            return df[df[col] > threshold]
        elif operation == 'sort':
            return df.sort_values(by=col, ascending=ascending)
        elif operation == 'mean':
            return df[col].mean()
        elif operation == 'std':
            return df[col].std()
        elif operation == 'var':
            return df[col].var()
        elif operation == 'correlation':
            return df[[col, col2]].corr().iloc[0, 1]
        elif operation == 'unique':
            return df[col].unique()
        elif operation == 'groupby':
            if agg_func == 'sum':
                return df.groupby(group_col)[agg_col].sum()
            elif agg_func == 'mean':
                return df.groupby(group_col)[agg_col].mean()
            elif agg_func == 'count':
                return df.groupby(group_col)[agg_col].count()
            elif agg_func == 'min':
                return df.groupby(group_col)[agg_col].min()
            elif agg_func == 'max':
                return df.groupby(group_col)[agg_col].max()

    # Use the provided number of cores or default to system cores
    if num_cores is None:
        num_cores = mproc.cpu_count()
    
    # Optimize chunk size based on operation type
    # For operations that benefit from larger chunks (like sorting), use larger chunks
    # For operations that are more CPU-bound (like min/max), use smaller chunks
    if operation in ['sort', 'groupby']:
        chunk_size = max(50000, len(df) // num_cores)
    else:
        chunk_size = max(10000, len(df) // num_cores)
    
    # For operations that can use numpy directly, convert to numpy arrays first
    if operation in ['min', 'max', 'sum', 'mean', 'std', 'var']:
        # Convert to numpy array once to avoid repeated conversions
        numpy_data = df[col].values
        chunks = [numpy_data[i:i + chunk_size] for i in range(0, len(numpy_data), chunk_size)]
    else:
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        if operation in ['min', 'max', 'sum']:
            # For min, max, sum operations, we already have numpy arrays
            if operation == 'min':
                results = list(executor.map(np.min, chunks))
                return min(results)
            elif operation == 'max':
                results = list(executor.map(np.max, chunks))
                return max(results)
            elif operation == 'sum':
                results = list(executor.map(np.sum, chunks))
                return sum(results)
        elif operation == 'count':
            results = list(executor.map(parallel_count, chunks))
            return sum(results)
        elif operation == 'search':
            results = list(executor.map(partial(parallel_search, col=col, value=value), chunks))
            return pd.concat(results, ignore_index=True)
        elif operation == 'filter':
            results = list(executor.map(partial(parallel_filter, col=col, threshold=threshold), chunks))
            return pd.concat(results, ignore_index=True)
        elif operation == 'sort':
            # For sorting, we need to sort each chunk and then merge them
            results = list(executor.map(partial(parallel_sort, col=col, ascending=ascending), chunks))
            # Merge sorted chunks
            return pd.concat(results, ignore_index=True)
        elif operation == 'mean':
            # For mean, we need to calculate sum and count for each chunk
            # Use numpy operations for better performance
            sum_results = list(executor.map(np.sum, chunks))
            count_results = list(executor.map(len, chunks))
            total_sum = sum(sum_results)
            total_count = sum(count_results)
            return total_sum / total_count
        elif operation == 'std':
            # For standard deviation, we need to calculate mean first
            mean = np.mean(df[col].values)
            # Then calculate sum of squared differences using numpy
            sum_sq_diff_results = list(executor.map(
                lambda x: np.sum((x - mean) ** 2), 
                chunks
            ))
            total_sum_sq_diff = sum(sum_sq_diff_results)
            total_count = len(df)
            return np.sqrt(total_sum_sq_diff / (total_count - 1))
        elif operation == 'var':
            # For variance, similar to std but without the square root
            mean = np.mean(df[col].values)
            sum_sq_diff_results = list(executor.map(
                lambda x: np.sum((x - mean) ** 2), 
                chunks
            ))
            total_sum_sq_diff = sum(sum_sq_diff_results)
            total_count = len(df)
            return total_sum_sq_diff / (total_count - 1)
        elif operation == 'correlation':
            # For correlation, we need to calculate covariance and variances
            # Convert to numpy arrays for better performance
            numpy_data1 = df[col].values
            numpy_data2 = df[col2].values
            mean1 = np.mean(numpy_data1)
            mean2 = np.mean(numpy_data2)
            
            # Create chunks for both columns
            chunks1 = [numpy_data1[i:i + chunk_size] for i in range(0, len(numpy_data1), chunk_size)]
            chunks2 = [numpy_data2[i:i + chunk_size] for i in range(0, len(numpy_data2), chunk_size)]
            
            # Calculate covariance
            cov_results = list(executor.map(
                lambda x, y: np.sum((x - mean1) * (y - mean2)), 
                chunks1, chunks2
            ))
            total_cov = sum(cov_results)
            
            # Calculate variances
            var1_results = list(executor.map(
                lambda x: np.sum((x - mean1) ** 2), 
                chunks1
            ))
            var2_results = list(executor.map(
                lambda x: np.sum((x - mean2) ** 2), 
                chunks2
            ))
            total_var1 = sum(var1_results)
            total_var2 = sum(var2_results)
            
            # Calculate correlation
            return total_cov / np.sqrt(total_var1 * total_var2)
        elif operation == 'unique':
            # For unique values, we need to get unique values from each chunk and then combine
            # Convert to numpy array for better performance
            numpy_data = df[col].values
            chunks = [numpy_data[i:i + chunk_size] for i in range(0, len(numpy_data), chunk_size)]
            results = list(executor.map(np.unique, chunks))
            # Combine all unique values and get unique again
            all_unique = np.unique(np.concatenate(results))
            return all_unique
        elif operation == 'groupby':
            # For groupby, we need to group by in each chunk and then combine
            results = list(executor.map(
                partial(parallel_groupby, group_col=group_col, agg_col=agg_col, agg_func=agg_func), 
                chunks
            ))
            # Combine results from all chunks
            combined = pd.concat(results)
            # Regroup to handle groups that might be split across chunks
            if agg_func == 'sum':
                return combined.groupby(level=0).sum()
            elif agg_func == 'mean':
                # For mean, we need to handle weighted averages
                if len(results) > 1:
                    # Calculate weighted mean based on counts
                    counts = pd.concat([r.groupby(level=0).count() for r in results]).groupby(level=0).sum()
                    weighted_sum = combined.groupby(level=0).sum()
                    return weighted_sum / counts
                return combined
            elif agg_func == 'count':
                return combined.groupby(level=0).sum()
            elif agg_func == 'min':
                return combined.groupby(level=0).min()
            elif agg_func == 'max':
                return combined.groupby(level=0).max()

def main():
    importlib.reload(mp)

    st.set_page_config(page_title="Efficient Data Analysis App", layout="wide")
    st.title("🔍 Efficient Data Analysis App")
    st.markdown(
        "Upload a CSV and explore **Sorting**, **Filtering**, **Searching**, **Grouping**, and **Parallel Operations** with timings."
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
            if st.checkbox("Step 1.1: Optimize memory usage of uploaded data"):
                data, start_mem, end_mem = reduce_memory_usage(df)
                st.success(f"Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({100*(start_mem - end_mem)/start_mem:.1f}% reduction).")
                fig, ax = plt.subplots()
                ax.bar(["Before", "After"], [start_mem, end_mem], color=["#FF6961", "#77DD77"])
                ax.set_title("Memory Optimization Impact")
                ax.set_ylabel("Memory (MB)")
                st.pyplot(fig)
            if not st.session_state['Run']:
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error: {e}")
            return

    
    

    # — Choose Operation —
    op = st.sidebar.selectbox(
        "🛠️ Select Operation",
        ["Sorting", "Filtering", "Searching", "Grouping", "Parallel Operations"]
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

    # 5) Parallel Operations
    elif op == "Parallel Operations":
        with st.expander("⚡ Parallel Operations", expanded=True):
            # Display system information
            st.sidebar.markdown("### System Information")
            total_cores = mproc.cpu_count()
            st.sidebar.info(f"Total available CPU cores: {total_cores}")
            
            # Allow user to change number of cores
            num_cores = st.sidebar.slider(
                "Number of cores to use",
                min_value=1,
                max_value=total_cores,
                value=total_cores,
                help="Adjust the number of CPU cores to use for parallel processing"
            )
            
            # Group operations by complexity
            basic_ops = ["min", "max", "sum", "count"]
            advanced_ops = ["mean", "std", "var"]
            search_ops = ["search", "filter", "unique"]
            complex_ops = ["sort", "correlation", "groupby"]
            
            operation_type = st.sidebar.selectbox(
                "Operation Type",
                ["Basic Operations", "Advanced Statistics", "Search & Filter", "Complex Operations"]
            )
            
            if operation_type == "Basic Operations":
                operation = st.sidebar.selectbox("Operation", basic_ops)
                if operation in basic_ops:
                    col = st.sidebar.selectbox("Column", df.columns)
            
            elif operation_type == "Advanced Statistics":
                operation = st.sidebar.selectbox("Operation", advanced_ops)
                if operation in advanced_ops:
                    col = st.sidebar.selectbox("Column", df.columns)
            
            elif operation_type == "Search & Filter":
                operation = st.sidebar.selectbox("Operation", search_ops)
                if operation in ["search", "filter", "unique"]:
                    col = st.sidebar.selectbox("Column", df.columns)
                
                if operation == "search":
                    vals = sorted(df[col].dropna().unique())
                    value = st.sidebar.selectbox("Value", vals)
                
                if operation == "filter":
                    thr = st.sidebar.number_input(
                        "Threshold",
                        float(df[col].min()),
                        float(df[col].max()),
                        float(df[col].min())
                    )
            
            elif operation_type == "Complex Operations":
                operation = st.sidebar.selectbox("Operation", complex_ops)
                
                if operation == "sort":
                    col = st.sidebar.selectbox("Sort by Column", df.columns)
                    ascending = st.sidebar.radio("Order", ["Ascending", "Descending"]) == "Ascending"
                
                elif operation == "correlation":
                    col = st.sidebar.selectbox("First Column", df.columns)
                    col2 = st.sidebar.selectbox("Second Column", [c for c in df.columns if c != col])
                
                elif operation == "groupby":
                    group_col = st.sidebar.selectbox("Group by Column", df.columns)
                    agg_col = st.sidebar.selectbox("Aggregate Column", df.columns)
                    agg_func = st.sidebar.selectbox("Aggregation Function", ["sum", "mean", "count", "min", "max"])
            
            if st.sidebar.button("Run Parallel Operation"):
                with st.spinner("Processing..."):
                    start_time = time.time()
                    
                    # Prepare parameters based on operation
                    params = {
                        "df": df.copy(),
                        "operation": operation,
                        "num_cores": num_cores
                    }
                    
                    # Add operation-specific parameters
                    if operation in ["min", "max", "sum", "mean", "std", "var", "search", "filter", "unique", "sort"]:
                        params["col"] = col
                    
                    if operation == "search":
                        params["value"] = value
                    
                    if operation == "filter":
                        params["threshold"] = thr
                    
                    if operation == "sort":
                        params["ascending"] = ascending
                    
                    if operation == "correlation":
                        params["col2"] = col2
                    
                    if operation == "groupby":
                        params["group_col"] = group_col
                        params["agg_col"] = agg_col
                        params["agg_func"] = agg_func
                    
                    # Execute the operation
                    result = execute_parallel_operation(**params)
                    execution_time = time.time() - start_time
                
                st.success(f"Operation completed in {execution_time:.4f}s using {num_cores} cores")
                st.metric("Time (s)", f"{execution_time:.4f}")
                
                # Display results based on operation type
                if operation in ["min", "max", "sum", "count", "mean", "std", "var"]:
                    st.metric(f"Result ({operation})", f"{result}")
                elif operation == "correlation":
                    st.metric(f"Correlation between {col} and {col2}", f"{result:.4f}")
                elif operation == "unique":
                    st.write(f"Found {len(result)} unique values")
                    st.write(result)
                elif operation == "groupby":
                    st.write("Group by Results:")
                    st.dataframe(result)
                else:
                    st.dataframe(result)
                    
                # Compare with pandas implementation
                st.subheader("Comparison with Pandas")
                start_time = time.time()
                
                # Pandas implementation based on operation
                if operation == "min":
                    pandas_result = df[col].min()
                elif operation == "max":
                    pandas_result = df[col].max()
                elif operation == "sum":
                    pandas_result = df[col].sum()
                elif operation == "count":
                    pandas_result = len(df)
                elif operation == "search":
                    pandas_result = df[df[col] == value]
                elif operation == "filter":
                    pandas_result = df[df[col] > thr]
                elif operation == "sort":
                    pandas_result = df.sort_values(by=col, ascending=ascending)
                elif operation == "mean":
                    pandas_result = df[col].mean()
                elif operation == "std":
                    pandas_result = df[col].std()
                elif operation == "var":
                    pandas_result = df[col].var()
                elif operation == "correlation":
                    pandas_result = df[[col, col2]].corr().iloc[0, 1]
                elif operation == "unique":
                    pandas_result = df[col].unique()
                elif operation == "groupby":
                    if agg_func == 'sum':
                        pandas_result = df.groupby(group_col)[agg_col].sum()
                    elif agg_func == 'mean':
                        pandas_result = df.groupby(group_col)[agg_col].mean()
                    elif agg_func == 'count':
                        pandas_result = df.groupby(group_col)[agg_col].count()
                    elif agg_func == 'min':
                        pandas_result = df.groupby(group_col)[agg_col].min()
                    elif agg_func == 'max':
                        pandas_result = df.groupby(group_col)[agg_col].max()
                
                pandas_time = time.time() - start_time
                
                st.write(f"Pandas implementation time: {pandas_time:.4f}s")
                
                # Display pandas results based on operation type
                if operation in ["min", "max", "sum", "count", "mean", "std", "var"]:
                    st.write(f"Pandas result: {pandas_result}")
                elif operation == "correlation":
                    st.write(f"Pandas correlation: {pandas_result:.4f}")
                elif operation == "unique":
                    st.write(f"Pandas found {len(pandas_result)} unique values")
                    st.write(pandas_result)
                elif operation == "groupby":
                    st.write("Pandas group by results:")
                    st.dataframe(pandas_result)
                else:
                    st.dataframe(pandas_result)
                
                speedup = pandas_time / execution_time
                st.metric("Speedup", f"{speedup:.2f}x")

if __name__ == "__main__":
    main()
