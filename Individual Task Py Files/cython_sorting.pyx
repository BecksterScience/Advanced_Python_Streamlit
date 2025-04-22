from libc.stdlib cimport malloc, free
cimport numpy as cnp
import numpy as np

def merge_sort_cython(cnp.ndarray arr, bint ascending=True):
    cdef int n = arr.shape[0]
    cdef cnp.ndarray temp = np.empty_like(arr)
    _merge_sort(arr, temp, 0, n - 1, ascending)

cdef void _merge_sort(cnp.ndarray arr, cnp.ndarray temp, int left, int right, bint ascending):
    if left < right:
        mid = (left + right) // 2
        _merge_sort(arr, temp, left, mid, ascending)
        _merge_sort(arr, temp, mid + 1, right, ascending)
        _merge(arr, temp, left, mid, right, ascending)

cdef void _merge(cnp.ndarray arr, cnp.ndarray temp, int left, int mid, int right, bint ascending):
    i, j, k = left, mid + 1, left
    while i <= mid and j <= right:
        if (ascending and arr[i] <= arr[j]) or (not ascending and arr[i] >= arr[j]):
            temp[k] = arr[i]
            i += 1
        else:
            temp[k] = arr[j]
            j += 1
        k += 1
    while i <= mid:
        temp[k] = arr[i]
        i += 1
        k += 1
    while j <= right:
        temp[k] = arr[j]
        j += 1
        k += 1
    for i in range(left, right + 1):
        arr[i] = temp[i]

def heap_sort_cython(cnp.ndarray arr, bint ascending=True):
    cdef int n = arr.shape[0]
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i, ascending)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        _heapify(arr, i, 0, ascending)

cdef void _heapify(cnp.ndarray arr, int n, int i, bint ascending):
    largest_or_smallest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and ((ascending and arr[left] > arr[largest_or_smallest]) or (not ascending and arr[left] < arr[largest_or_smallest])):
        largest_or_smallest = left

    if right < n and ((ascending and arr[right] > arr[largest_or_smallest]) or (not ascending and arr[right] < arr[largest_or_smallest])):
        largest_or_smallest = right

    if largest_or_smallest != i:
        arr[i], arr[largest_or_smallest] = arr[largest_or_smallest], arr[i]
        _heapify(arr, n, largest_or_smallest, ascending)

def selection_sort_cython(cnp.ndarray arr, bint ascending=True):
    cdef int n = arr.shape[0]
    cdef int i, j, idx
    for i in range(n):
        idx = i
        for j in range(i + 1, n):
            if (ascending and arr[j] < arr[idx]) or (not ascending and arr[j] > arr[idx]):
                idx = j
        arr[i], arr[idx] = arr[idx], arr[i]

def timsort_cython(cnp.ndarray arr, bint ascending=True):
    """
    Cython implementation of Timsort using Python's built-in sorted function.
    """
    cdef list py_list = arr.tolist()
    py_list = sorted(py_list, reverse=not ascending)
    return np.array(py_list, dtype=arr.dtype)
