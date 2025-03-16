import time
import pandas as pd
import numpy as np
from cython_bubble_sort import bubble_sort_cython
from cython_quicksort import quicksort_cython
from numba import njit

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

if __name__ == '__main__':
    df = pd.DataFrame(np.random.randint(0, 100, size=(1000, 4)), columns=list('ABCD'))