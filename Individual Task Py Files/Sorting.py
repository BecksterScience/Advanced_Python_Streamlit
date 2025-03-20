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
                if df[col_name][j] > df[col_name][j+1]:
                    df.loc[j], df.loc[j+1] = df.loc[j+1].copy(), df.loc[j].copy()
            else:
                if df[col_name][j] < df[col_name][j+1]:
                    df.loc[j], df.loc[j+1] = df.loc[j+1].copy(), df.loc[j].copy()
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

def bubble_sort_numpy(df, col_name, ascending=True):
    '''
    Sorts the dataframe based on the column name in ascending or descending order using numpy.
    :param df: pandas dataframe
    :param col_name: column name on which sorting needs to be done
    :param ascending: boolean value to sort in ascending or descending order
    :return: sorted dataframe, time taken
    '''
    start_time = time.time()
    numpy_array = df[col_name].to_numpy()
    col_index = 0  # Since we are only sorting one column, its index is 0
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array)-1):
            if ascending:
                if numpy_array[j] > numpy_array[j+1]:
                    numpy_array[j], numpy_array[j+1] = numpy_array[j+1].copy(), numpy_array[j].copy()
            else:
                if numpy_array[j] < numpy_array[j+1]:
                    numpy_array[j], numpy_array[j+1] = numpy_array[j+1].copy(), numpy_array[j].copy()
    df[col_name] = numpy_array
    end_time = time.time()
    time_taken = end_time - start_time
    return df, time_taken

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

if __name__ == '__main__':
    df = pd.DataFrame(np.random.randint(0, 100, size=(1000, 4)), columns=list('ABCD'))