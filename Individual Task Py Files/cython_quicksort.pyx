import numpy as np
cimport numpy as np

def quicksort_cython(np.ndarray numpy_array, int low, int high, bint ascending=True):
    if low < high:
        pi = partition(numpy_array, low, high, ascending)
        quicksort_cython(numpy_array, low, pi - 1, ascending)
        quicksort_cython(numpy_array, pi + 1, high, ascending)

cdef int partition(np.ndarray numpy_array, int low, int high, bint ascending):
    cdef double pivot = numpy_array[high, 0]
    cdef int i = low - 1
    cdef int j
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