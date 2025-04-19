import time
import pandas as pd
import numpy as np
from cython_bubble_sort import bubble_sort_cython
from cython_quicksort import quicksort_cython
from numba import njit
from cython_sorting import merge_sort_cython, heap_sort_cython

def bubble_sort(df, col_name, ascending=True):
    """
    Sorts the dataframe based on the column name in ascending or descending order.
    :param df: pandas dataframe
    :param col_name: column name on which sorting needs to be done
    :param ascending: boolean value to sort in ascending or descending order
    :return: sorted dataframe, time taken
    """
    start_time = time.time()
    for i in range(len(df)):
        for j in range(len(df)-1):
            if ascending:
                if df[col_name].iloc[j] > df[col_name].iloc[j+1]:
                    df.iloc[j], df.iloc[j+1] = df.iloc[j+1].copy(), df.iloc[j].copy()
            else:
                if df[col_name].iloc[j] < df[col_name].iloc[j+1]:
                    df.iloc[j], df.iloc[j+1] = df.iloc[j+1].copy(), df.iloc[j].copy()
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def bubble_sort_numpy(df, col_name, ascending=True):
    start_time = time.time()
    df = df.copy()

    arr = df[col_name].to_numpy()
    idx = np.arange(len(df))  # track row positions

    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if (ascending and arr[j] > arr[j + 1]) or (not ascending and arr[j] < arr[j + 1]):
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                idx[j], idx[j + 1] = idx[j + 1], idx[j]

    df_sorted = df.iloc[idx].reset_index(drop=True)
    end_time = time.time()
    return df_sorted, end_time - start_time

def bubble_sort_cython_wrapper(df, col_name, ascending=True):
    '''
    Sorts the dataframe based on the column name in ascending or descending order using Cython.
    :param df: pandas dataframe
    :param col_name: column name on which sorting needs to be done
    :param ascending: boolean value to sort in ascending or descending order
    :return: sorted dataframe, time taken
    '''
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    col_index = 0  # Since we are only sorting one column, its index is 0
    
    # Check if the column contains numeric data
    if pd.api.types.is_numeric_dtype(df[col_name]):
        numpy_array = numpy_array.astype(np.float64)
    else:
        numpy_array = numpy_array.astype(object)
    
    sorted_array = bubble_sort_cython(numpy_array.reshape(-1, 1), col_index, ascending)
    df[col_name] = sorted_array.flatten()
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

@njit
def bubble_sort_numba_impl(numpy_array, col_index, ascending):
    for i in range(numpy_array.shape[0]):
        for j in range(numpy_array.shape[0] - 1):
            if ascending:
                if numpy_array[j, col_index] > numpy_array[j + 1, col_index]:
                    numpy_array[j], numpy_array[j + 1] = numpy_array[j + 1].copy(), numpy_array[j].copy()
            else:
                if numpy_array[j, col_index] < numpy_array[j + 1, col_index]:
                    numpy_array[j], numpy_array[j + 1] = numpy_array[j + 1].copy(), numpy_array[j].copy()
    return numpy_array

def bubble_sort_numba(df, col_name, ascending=True):
    '''
    Sorts the dataframe based on the column name in ascending or descending order using Numba.
    :param df: pandas dataframe
    :param col_name: column name on which sorting needs to be done
    :param ascending: boolean value to sort in ascending or descending order
    :return: sorted dataframe, time taken
    '''
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    col_index = 0  # Since we are only sorting one column, its index is 0
    sorted_array = bubble_sort_numba_impl(numpy_array.reshape(-1, 1), col_index, ascending)
    df[col_name] = sorted_array.flatten()
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def quicksort(arr, low, high, ascending=True):
    if low < high:
        pi = partition(arr, low, high, ascending)
        quicksort(arr, low, pi - 1, ascending)
        quicksort(arr, pi + 1, high, ascending)

def partition(arr, low, high, ascending):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if ascending:
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        else:
            if arr[j] > pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quicksort_python(df, col_name, ascending=True):
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    quicksort(numpy_array, 0, len(numpy_array) - 1, ascending)
    df[col_name] = numpy_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def quicksort_numpy(df, col_name, ascending=True):
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    numpy_array.sort(kind='quicksort')
    if not ascending:
        numpy_array = numpy_array[::-1]
    df[col_name] = numpy_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def quicksort_cython_wrapper(df, col_name, ascending=True):
    start_time = time.time()
    numpy_array = df[col_name].to_numpy().reshape(-1, 1)
    col_index = 0  # Since we are only sorting one column, its index is 0
    
    # Check if the column contains numeric data
    if pd.api.types.is_numeric_dtype(df[col_name]):
        numpy_array = numpy_array.astype(np.float64)
    
    quicksort_cython(numpy_array, 0, len(numpy_array) - 1, ascending)
    df[col_name] = numpy_array.flatten()
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

@njit
def quicksort_numba_impl(numpy_array, low, high, ascending):
    if low < high:
        pi = partition_numba(numpy_array, low, high, ascending)
        quicksort_numba_impl(numpy_array, low, pi - 1, ascending)
        quicksort_numba_impl(numpy_array, pi + 1, high, ascending)

@njit
def partition_numba(numpy_array, low, high, ascending):
    pivot = numpy_array[high, 0]
    i = low - 1
    for j in range(low, high):
        if ascending:
            if numpy_array[j, 0] < pivot:
                i += 1
                numpy_array[i], numpy_array[j] = numpy_array[j].copy(), numpy_array[i].copy()
        else:
            if numpy_array[j, 0] > pivot:
                i += 1
                numpy_array[i], numpy_array[j] = numpy_array[j].copy(), numpy_array[i].copy()
    numpy_array[i + 1], numpy_array[high] = numpy_array[high].copy(), numpy_array[i + 1].copy()
    return i + 1

def quicksort_numba(df, col_name, ascending=True):
    start_time = time.time()
    numpy_array = df[col_name].to_numpy().reshape(-1, 1)
    quicksort_numba_impl(numpy_array, 0, len(numpy_array) - 1, ascending)
    df[col_name] = numpy_array.flatten()
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def merge_sort(arr, ascending=True):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half, ascending)
        merge_sort(right_half, ascending)

        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if (ascending and left_half[i] < right_half[j]) or (not ascending and left_half[i] > right_half[j]):
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

def merge_sort_python(df, col_name, ascending=True):
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    merge_sort(numpy_array, ascending)
    df[col_name] = numpy_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def merge_sort_numpy(df, col_name, ascending=True):
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    numpy_array.sort(kind='mergesort')
    if not ascending:
        numpy_array = numpy_array[::-1]
    df[col_name] = numpy_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def merge_sort_cython_wrapper(df, col_name, ascending=True):
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    merge_sort_cython(numpy_array, ascending)
    df[col_name] = numpy_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

@njit
def merge_sort_numba_impl(arr, ascending):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort_numba_impl(left_half, ascending)
        merge_sort_numba_impl(right_half, ascending)

        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if (ascending and left_half[i] < right_half[j]) or (not ascending and left_half[i] > right_half[j]):
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

def merge_sort_numba(df, col_name, ascending=True):
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    merge_sort_numba_impl(numpy_array, ascending)
    df[col_name] = numpy_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def heapify(arr, n, i, ascending):
    largest_or_smallest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and ((ascending and arr[left] > arr[largest_or_smallest]) or (not ascending and arr[left] < arr[largest_or_smallest])):
        largest_or_smallest = left

    if right < n and ((ascending and arr[right] > arr[largest_or_smallest]) or (not ascending and arr[right] < arr[largest_or_smallest])):
        largest_or_smallest = right

    if largest_or_smallest != i:
        arr[i], arr[largest_or_smallest] = arr[largest_or_smallest], arr[i]
        heapify(arr, n, largest_or_smallest, ascending)

def heap_sort(arr, ascending=True):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i, ascending)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0, ascending)

def heap_sort_python(df, col_name, ascending=True):
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    heap_sort(numpy_array, ascending)
    df[col_name] = numpy_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def heap_sort_numpy(df, col_name, ascending=True):
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    numpy_array.sort(kind='heapsort')
    if not ascending:
        numpy_array = numpy_array[::-1]
    df[col_name] = numpy_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def heap_sort_cython_wrapper(df, col_name, ascending=True):
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    heap_sort_cython(numpy_array, ascending)
    df[col_name] = numpy_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

@njit
def heapify_numba(arr, n, i, ascending):
    largest_or_smallest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and ((ascending and arr[left] > arr[largest_or_smallest]) or (not ascending and arr[left] < arr[largest_or_smallest])):
        largest_or_smallest = left

    if right < n and ((ascending and arr[right] > arr[largest_or_smallest]) or (not ascending and arr[right] < arr[largest_or_smallest])):
        largest_or_smallest = right

    if largest_or_smallest != i:
        arr[i], arr[largest_or_smallest] = arr[largest_or_smallest], arr[i]
        heapify_numba(arr, n, largest_or_smallest, ascending)

@njit
def heap_sort_numba_impl(arr, ascending):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify_numba(arr, n, i, ascending)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify_numba(arr, i, 0, ascending)

def heap_sort_numba(df, col_name, ascending=True):
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    heap_sort_numba_impl(numpy_array, ascending)
    df[col_name] = numpy_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def selection_sort(arr, ascending=True):
    n = len(arr)
    for i in range(n):
        idx = i
        for j in range(i + 1, n):
            if (ascending and arr[j] < arr[idx]) or (not ascending and arr[j] > arr[idx]):
                idx = j
        arr[i], arr[idx] = arr[idx], arr[i]

def selection_sort_python(df, col_name, ascending=True):
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    selection_sort(numpy_array, ascending)
    df[col_name] = numpy_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def selection_sort_numpy(df, col_name, ascending=True):
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    for i in range(len(numpy_array)):
        idx = i + np.argmin(numpy_array[i:] if ascending else -numpy_array[i:])
        numpy_array[i], numpy_array[idx] = numpy_array[idx], numpy_array[i]
    df[col_name] = numpy_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

@njit
def selection_sort_numba_impl(arr, ascending):
    n = len(arr)
    for i in range(n):
        idx = i
        for j in range(i + 1, n):
            if (ascending and arr[j] < arr[idx]) or (not ascending and arr[j] > arr[idx]):
                idx = j
        arr[i], arr[idx] = arr[idx], arr[i]

def selection_sort_numba(df, col_name, ascending=True):
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    selection_sort_numba_impl(numpy_array, ascending)
    df[col_name] = numpy_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def selection_sort_cython_wrapper(df, col_name, ascending=True):
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    from cython_sorting import selection_sort_cython
    selection_sort_cython(numpy_array, ascending)
    df[col_name] = numpy_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def timsort_python(df, col_name, ascending=True):
    """
    Sorts the dataframe using Python's built-in Timsort.
    :param df: pandas dataframe
    :param col_name: column name on which sorting needs to be done
    :param ascending: boolean value to sort in ascending or descending order
    :return: sorted dataframe, time taken
    """
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    sorted_array = sorted(numpy_array, reverse=not ascending)
    df[col_name] = sorted_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def timsort_numpy(df, col_name, ascending=True):
    """
    Sorts the dataframe using NumPy's Timsort.
    :param df: pandas dataframe
    :param col_name: column name on which sorting needs to be done
    :param ascending: boolean value to sort in ascending or descending order
    :return: sorted dataframe, time taken
    """
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    numpy_array.sort(kind='stable')  # NumPy's stable sort uses Timsort
    if not ascending:
        numpy_array = numpy_array[::-1]
    df[col_name] = numpy_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

@njit
def timsort_numba_impl(arr, ascending):
    """
    Numba implementation of Timsort using Python's built-in sorted function.
    """
    sorted_array = sorted(arr, reverse=not ascending)
    return sorted_array

def timsort_numba(df, col_name, ascending=True):
    """
    Sorts the dataframe using Numba-accelerated Timsort.
    :param df: pandas dataframe
    :param col_name: column name on which sorting needs to be done
    :param ascending: boolean value to sort in ascending or descending order
    :return: sorted dataframe, time taken
    """
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    sorted_array = timsort_numba_impl(numpy_array, ascending)
    df[col_name] = sorted_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def timsort_cython_wrapper(df, col_name, ascending=True):
    """
    Sorts the dataframe using Cython-accelerated Timsort.
    :param df: pandas dataframe
    :param col_name: column name on which sorting needs to be done
    :param ascending: boolean value to sort in ascending or descending order
    :return: sorted dataframe, time taken
    """
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    from cython_sorting import timsort_cython
    sorted_array = timsort_cython(numpy_array, ascending)
    df[col_name] = sorted_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

SORTING_ALGORITHMS = {
    "Bubble Sort (Python)": bubble_sort,
    "Bubble Sort (NumPy)": bubble_sort_numpy,
    "Bubble Sort (Cython)": bubble_sort_cython_wrapper,
    "Bubble Sort (Numba)": bubble_sort_numba,
    "Quick Sort (Python)": quicksort_python,
    "Quick Sort (NumPy)": quicksort_numpy,
    "Quick Sort (Cython)": quicksort_cython_wrapper,
    "Quick Sort (Numba)": quicksort_numba,
    "Merge Sort (Python)": merge_sort_python,
    "Merge Sort (NumPy)": merge_sort_numpy,
    "Merge Sort (Cython)": merge_sort_cython_wrapper,
    "Merge Sort (Numba)": merge_sort_numba,
    "Heap Sort (Python)": heap_sort_python,
    "Heap Sort (NumPy)": heap_sort_numpy,
    "Heap Sort (Cython)": heap_sort_cython_wrapper,
    "Heap Sort (Numba)": heap_sort_numba,
    "Selection Sort (Python)": selection_sort_python,
    "Selection Sort (NumPy)": selection_sort_numpy,
    "Selection Sort (Cython)": selection_sort_cython_wrapper,
    "Selection Sort (Numba)": selection_sort_numba,
    "Timsort (Python)": timsort_python,
    "Timsort (NumPy)": timsort_numpy,
    "Timsort (Cython)": timsort_cython_wrapper,
    "Timsort (Numba)": timsort_numba,
}

def filter_baseline(file_path, filter_value):
    df = pd.read_csv(file_path)
    filtered_df = df[df['int_col'] > filter_value]
    return filtered_df

# Optimized filtering function using Pandas
def filter_data_pandas(file_path, filter_value):
    df = pd.read_csv(file_path)
    filtered_df = df.loc[df['int_col'] > filter_value]
    return filtered_df

# Optimized filtering function using Numpy
def filter_data_numpy(file_path, filter_value):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    filtered_data = data[data[:, 0] > filter_value]
    return filtered_data

# Optimized filtering function using Numba
@njit
def filter_data_numba(data, filter_value):
    filtered_data = []
    for row in data:
        if row[0] > filter_value:
            filtered_data.append(row)
    return filtered_data

def filter_data_numba_wrapper(file_path, filter_value):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    return filter_data_numba(data, filter_value)

def search_baseline(file_path, search_value):
    df = pd.read_csv(file_path)
    search_results = df[df['str_col'] == search_value]
    return search_results

# Optimized searching function using Pandas
def search_data_pandas(file_path, search_value):
    df = pd.read_csv(file_path)
    search_results = df.loc[df['str_col'] == search_value]
    return search_results

# Optimized searching function using NumPy
def search_data_numpy(file_path, search_value):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype=str)
    search_results = data[data[:, 2] == search_value]
    return search_results

# Optimized searching function using string functions
def search_optimization(file_path, search_value):
    df = pd.read_csv(file_path)
    search_results = df[df['str_col'].str.contains(search_value)]
    return search_results

def group_and_aggregate_normal(data, group_columns, agg_dict):
    """
    Normal grouping and aggregation using pandas groupby.
    
    Parameters:
    - data: pandas DataFrame.
    - group_columns: list of columns to group by.
    - agg_dict: dictionary mapping column names to aggregation functions.
    
    Returns:
    - Aggregated DataFrame and time elapsed.
    """
    start_time = time.time()
    result = data.groupby(group_columns).agg(agg_dict)
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def group_and_aggregate_chunk(data_chunk, group_columns, agg_dict, is_mean=False):
    """
    Process a chunk of data by grouping and aggregating.
    
    If is_mean is False, simply apply the aggregation.
    If is_mean is True, compute both sum and count for each column (to later calculate mean).
    """
    if not is_mean:
        return data_chunk.groupby(group_columns).agg(agg_dict)
    else:
        # For mean, compute both sum and count for later combining.
        agg_operations = {col: ['sum', 'count'] for col in agg_dict.keys()}
        return data_chunk.groupby(group_columns).agg(agg_operations)

def combine_non_mean(partial_results, agg_func, group_columns):
    """
    Combine partial results for aggregations that are directly mergeable (sum, min, max, count).
    """
    combined = pd.concat(partial_results)
    return combined.groupby(level=group_columns).agg(agg_func)

def combine_mean(partial_results, agg_columns, group_columns):
    """
    Combine partial results for the 'mean' aggregation.
    
    Each partial result contains multi-level columns with ('col', 'sum') and ('col', 'count').
    The function sums up the partial sums and counts, then computes the final mean.
    """
    combined = pd.concat(partial_results)
    combined_grouped = combined.groupby(level=group_columns).sum()  # Sum partial sums and counts
    mean_dict = {}
    for col in agg_columns:
        mean_dict[col] = combined_grouped[(col, 'sum')] / combined_grouped[(col, 'count')]
    result = pd.DataFrame(mean_dict)
    result.index.names = group_columns
    return result

def group_and_aggregate_optimized(data, group_columns, agg_func, agg_columns, chunk_size=10000):
    """
    Optimized grouping and aggregation using parallel processing.
    
    Parameters:
    - data: pandas DataFrame.
    - group_columns: list of columns to group by.
    - agg_func: aggregation function as string ('sum', 'min', 'max', 'count', or 'mean').
    - agg_columns: list of numeric columns to aggregate.
    - chunk_size: number of rows per chunk.
    
    Returns:
    - Aggregated DataFrame and time elapsed.
    """
    is_mean = (agg_func == "mean")
    if not is_mean:
        agg_dict = {col: agg_func for col in agg_columns}
    else:
        # For mean, we'll compute sum and count for each column.
        agg_dict = {col: None for col in agg_columns}

    # Split data into chunks.
    chunks = []
    n = len(data)
    for i in range(0, n, chunk_size):
        chunks.append(data.iloc[i:i+chunk_size])
    
    start_time = time.time()
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(group_and_aggregate_chunk, chunk, group_columns, agg_dict, is_mean) for chunk in chunks]
        partial_results = [future.result() for future in futures]
    
    if not is_mean:
        optimized_result = combine_non_mean(partial_results, agg_func, group_columns)
    else:
        optimized_result = combine_mean(partial_results, agg_columns, group_columns)
    
    elapsed_time = time.time() - start_time
    return optimized_result, elapsed_time

def time_function(func, *args, runs=5):
    """
    Execute a function multiple times and return its result along with the average runtime.
    
    Parameters:
        func: function to run.
        *args: arguments to pass to func.
        runs (int): number of times to run func (default 5).
    
    Returns:
        result: result from last run.
        avg_time: average runtime (in seconds).
    """
    times = []
    result = None
    for _ in range(runs):
        start = time.time()
        result = func(*args)
        times.append(time.time() - start)
    return result, np.mean(times)

# ----------------------------
# GROUPING FUNCTIONS
# ----------------------------

# Group by Single Field (Count)
def group_by_single_field_vectorized(data, group_col):
    """Vectorized method using Pandas groupby to count occurrences."""
    start = time.time()
    result = data.groupby(group_col)[group_col].count()
    elapsed = time.time() - start
    return result, elapsed

def group_by_single_field_loop(data, group_col):
    """Loop-based method (using iterrows) to count occurrences."""
    start = time.time()
    counts = {}
    for _, row in data.iterrows():
        key = row[group_col]
        counts[key] = counts.get(key, 0) + 1
    elapsed = time.time() - start
    return pd.Series(counts), elapsed

@njit
def numba_count(arr, max_val):
    """Numba function to count occurrences in an array assuming integer keys."""
    counts = np.zeros(max_val, dtype=np.int64)
    for x in arr:
        counts[x] += 1
    return counts

def group_by_single_field_numba(data, group_col):
    """
    Numba-accelerated method for counting occurrences.
    Note: group_col values must be convertible to int32.
    """
    arr = data[group_col].astype(np.int32).values
    max_val = int(arr.max()) + 1
    start = time.time()
    counts = numba_count(arr, max_val)
    elapsed = time.time() - start
    result = {i: int(counts[i]) for i in range(max_val) if counts[i] != 0}
    return result, elapsed

# Group by Multiple Fields (Mean)
def group_by_multiple_fields_vectorized(data, group_cols, agg_col):
    """Vectorized: Group by multiple fields and compute the mean using Pandas."""
    start = time.time()
    result = data.groupby(group_cols, as_index=False)[agg_col].mean()
    elapsed = time.time() - start
    return result, elapsed

def group_by_multiple_fields_loop(data, group_cols, agg_col):
    """Loop-based: Group by multiple fields using iterrows and compute the mean."""
    start = time.time()
    groups = {}
    for _, row in data.iterrows():
        key = tuple(row[g] for g in group_cols)
        groups.setdefault(key, []).append(row[agg_col])
    result = {key: np.mean(vals) for key, vals in groups.items()}
    elapsed = time.time() - start
    result_df = pd.DataFrame(list(result.items()), columns=["Group", "Mean"])
    return result_df, elapsed

# Group by Computed Column (e.g., Date to Decade Bucket)
def group_by_computed_column_vectorized(data, date_col, numeric_col):
    """
    Vectorized: Convert a date column to datetime and then group by a computed decade bucket.
    The bucket is computed as floor((year - 1900) / 10).
    """
    try:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    except Exception as e:
        print(f"Error converting {date_col}: {e}")
        return None, None
    bucket = np.floor((data[date_col].dt.year - 1900) / 10)
    start = time.time()
    result = data.groupby(bucket)[numeric_col].mean()
    elapsed = time.time() - start
    return result, elapsed

def group_by_computed_column_loop(data, date_col, numeric_col):
    """
    Loop-based: Convert the date column to datetime, compute the bucket for each row,
    then compute the mean for each bucket.
    """
    try:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    except Exception as e:
        print(f"Error converting {date_col}: {e}")
        return None, None
    buckets = np.floor((data[date_col].dt.year - 1900) / 10)
    start = time.time()
    groups = {}
    for i, b in enumerate(buckets):
        groups.setdefault(b, []).append(data.iloc[i][numeric_col])
    result = {b: np.mean(vals) for b, vals in groups.items()}
    elapsed = time.time() - start
    return pd.Series(result), elapsed

# Multiple Aggregates on One Column (Count and Mean)
def group_multiple_aggregates_vectorized(data, group_col, count_col, numeric_col):
    """Vectorized: Use Pandas agg to compute count and mean grouped by group_col."""
    start = time.time()
    result = data.groupby(group_col).agg({count_col: "count", numeric_col: "mean"})
    result = result.rename(columns={count_col: "NUM", numeric_col: "AVG"})
    elapsed = time.time() - start
    return result, elapsed

def group_multiple_aggregates_loop(data, group_col, count_col, numeric_col):
    """Loop-based: Aggregate count and mean for each group by iterating over rows."""
    start = time.time()
    groups = {}
    for _, row in data.iterrows():
        key = row[group_col]
        if key not in groups:
            groups[key] = {"count": 0, "sum": 0.0}
        groups[key]["count"] += 1
        groups[key]["sum"] += row[numeric_col]
    result = {key: {"NUM": val["count"], "AVG": val["sum"] / val["count"]} 
              for key, val in groups.items()}
    elapsed = time.time() - start
    result_df = pd.DataFrame.from_dict(result, orient="index")
    return result_df, elapsed

# Transform Grouping (Using transform to copy group aggregate to each row)
def transform_grouping_vectorized(data, group_col, numeric_col):
    """Use Pandas transform to compute the mean per group and assign it to each row."""
    start = time.time()
    data["TRANSFORMED"] = data.groupby(group_col)[numeric_col].transform("mean")
    elapsed = time.time() - start
    return data, elapsed

# Sum Aggregation Comparison
def vectorized_group_sum(data, group_col, val_col):
    """Vectorized: Aggregate sum using Pandas groupby."""
    return data.groupby(group_col)[val_col].sum()

def loop_group_sum(data, group_col, val_col):
    """Loop-based: Aggregate sum by iterating over rows."""
    result = {}
    for _, row in data.iterrows():
        key = row[group_col]
        result[key] = result.get(key, 0) + row[val_col]
    return result

@njit
def numba_loop_group_sum(groups, values, n_groups):
    """Numba-accelerated sum aggregation. Expects numeric grouping keys."""
    sums = np.zeros(n_groups)
    for i in range(groups.shape[0]):
        sums[groups[i]] += values[i]
    return sums

# ----------------------------
# FILTERING FUNCTIONS
# ----------------------------
def filter_by_threshold_vectorized(data, numeric_col, threshold):
    """Vectorized: Filter rows where data[numeric_col] > threshold using Pandas."""
    start = time.time()
    result = data[data[numeric_col] > threshold]
    elapsed = time.time() - start
    return result, elapsed

def filter_by_threshold_loop(data, numeric_col, threshold):
    """Loop-based: Filter rows using iterrows."""
    start = time.time()
    filtered_rows = []
    for _, row in data.iterrows():
        if row[numeric_col] > threshold:
            filtered_rows.append(row)
    result = pd.DataFrame(filtered_rows)
    elapsed = time.time() - start
    return result, elapsed

@njit
def numba_filter_indices(values, threshold):
    """
    Numba function: returns indices where values > threshold.
    Preallocates an output array and trims it.
    """
    n = values.shape[0]
    out = np.empty(n, dtype=np.int64)
    count = 0
    for i in range(n):
        if values[i] > threshold:
            out[count] = i
            count += 1
    return out[:count]

def filter_by_threshold_numba(data, numeric_col, threshold):
    """
    Numba-accelerated filtering: convert numeric_col to a NumPy array and use Numba to get indices.
    """
    values = data[numeric_col].values.astype(np.float64)
    start = time.time()
    indices = numba_filter_indices(values, threshold)
    result = data.iloc[indices]
    elapsed = time.time() - start
    return result, elapsed

# ----------------------------
# SEARCHING FUNCTIONS
# ----------------------------
def search_substring_vectorized(data, text_col, substring):
    """Vectorized: Search for substring in a text column using .str.contains."""
    start = time.time()
    result = data[data[text_col].str.contains(substring, na=False)]
    elapsed = time.time() - start
    return result, elapsed

def search_substring_loop(data, text_col, substring):
    """Loop-based: Iterate over rows and select rows where substring is in the text column."""
    start = time.time()
    found_rows = []
    for _, row in data.iterrows():
        cell = str(row[text_col]) if pd.notnull(row[text_col]) else ""
        if substring in cell:
            found_rows.append(row)
    result = pd.DataFrame(found_rows)
    elapsed = time.time() - start
    return result, elapsed

# ----------------------------
# ADVANCED GROUPING FUNCTIONS
# ----------------------------
def alignment_grouping(data, group_col, base_list, agg_col):
    """
    Alignment Grouping:
    Ensures each element of base_list appears in the result by merging data with a base DataFrame and then grouping.
    """
    base_df = pd.DataFrame({group_col: base_list})
    merged = pd.merge(base_df, data, on=group_col, how="left")
    result = merged.groupby(group_col).agg({agg_col: "count"})
    # Adjust counts: subtract the extra row from the base DataFrame if present.
    result = result.applymap(lambda x: x - 1 if x > 0 else 0)
    return result

def enumeration_grouping(data, num_col, conditions, labels):
    """
    Enumeration Grouping:
    For each condition (a function) in the provided list,
    count the number of records (based on num_col) that satisfy that condition.
    
    Parameters:
        data: DataFrame.
        num_col: the column on which to evaluate the condition.
        conditions: a list of functions that take a value and return a boolean.
        labels: a list of corresponding labels.
    
    Returns:
        A DataFrame with labels and counts.
    """
    results = []
    for cond, label in zip(conditions, labels):
        subset = data[data[num_col].apply(cond)]
        results.append((label, subset.shape[0]))
    return pd.DataFrame(results, columns=[num_col, "count"])

# ----------------------------
# GROUP BY CHANGED VALUE (ADVANCED)
# ----------------------------
def group_by_changed_value(data, col):
    """
    Group by Changed Value:
    Group rows based on when the value in the specified column changes.
    A new group is identified whenever the value changes.
    """
    start = time.time()
    grouping_key = (data[col] != data[col].shift()).cumsum()
    result = data.groupby(grouping_key).apply(lambda x: x.sort_values(col))
    elapsed = time.time() - start
    return result, elapsed

if __name__ == '__main__':
    # Create the DataFrame to sort
    df = pd.DataFrame(np.random.randint(0, 100, size=(1000, 4)), columns=list('ABCD'))
    col_to_sort = 'A'

    print(f"\nSorting DataFrame by column '{col_to_sort}' using all sorting algorithms:\n")

    for name, sort_func in SORTING_ALGORITHMS.items():
        try:
            sorted_df, elapsed = sort_func(df.copy(), col_to_sort)
            print(f"--- {name} ---")
            print(sorted_df.head(5))  # Show top 5 rows
            print(f"Time taken: {elapsed:.6f} seconds\n")
        except Exception as e:
            print(f"--- {name} FAILED ---")
            print(f"Error: {e}\n")