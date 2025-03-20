import numpy as np
cimport numpy as np

def quicksort_cython(np.ndarray numpy_array, int low, int high, bint ascending=True):
    cdef int pi
    if low < high:
        pi = partition(numpy_array, low, high, ascending)
        quicksort_cython(numpy_array, low, pi - 1, ascending)
        quicksort_cython(numpy_array, pi + 1, high, ascending)

cdef int partition(np.ndarray numpy_array, int low, int high, bint ascending):
    cdef double pivot = numpy_array[high, 0]
    cdef int i = low - 1
    cdef int j
    cdef double temp
    for j in range(low, high):
        if ascending:
            if numpy_array[j, 0] < pivot:
                i += 1
                temp = numpy_array[i, 0]
                numpy_array[i, 0] = numpy_array[j, 0]
                numpy_array[j, 0] = temp
        else:
            if numpy_array[j, 0] > pivot:
                i += 1
                temp = numpy_array[i, 0]
                numpy_array[i, 0] = numpy_array[j, 0]
                numpy_array[j, 0] = temp
    temp = numpy_array[i + 1, 0]
    numpy_array[i + 1, 0] = numpy_array[high, 0]
    numpy_array[high, 0] = temp
    return i + 1