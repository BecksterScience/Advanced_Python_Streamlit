import numpy as np
cimport numpy as np

def bubble_sort_cython(np.ndarray numpy_array, int col_index, bint ascending=True):
    cdef int i, j
    for i in range(numpy_array.shape[0]):
        for j in range(numpy_array.shape[0] - 1):
            if ascending:
                if numpy_array[j, col_index] > numpy_array[j + 1, col_index]:
                    numpy_array[j], numpy_array[j + 1] = numpy_array[j + 1].copy(), numpy_array[j].copy()
            else:
                if numpy_array[j, col_index] < numpy_array[j + 1, col_index]:
                    numpy_array[j], numpy_array[j + 1] = numpy_array[j + 1].copy(), numpy_array[j].copy()
    return numpy_array