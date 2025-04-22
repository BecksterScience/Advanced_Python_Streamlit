import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Individual Task Py Files')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from Sorting import (
    bubble_sort, bubble_sort_numpy, bubble_sort_cython_wrapper, bubble_sort_numba,
    quicksort_python, quicksort_numpy, quicksort_cython_wrapper, quicksort_numba,
    merge_sort_python, merge_sort_numpy, merge_sort_cython_wrapper, merge_sort_numba,
    heap_sort_python, heap_sort_numpy, heap_sort_cython_wrapper, heap_sort_numba,
    selection_sort_python, selection_sort_numpy, selection_sort_cython_wrapper, selection_sort_numba,
    timsort_python, timsort_numpy, timsort_cython_wrapper, timsort_numba
)

st.title("DataFrame Sorting Application")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("DataFrame:")
    st.write(df)

    col_name = st.selectbox("Select column to sort by", df.columns)
    ascending = st.radio("Select sorting order", ("Ascending", "Descending")) == "Ascending"
    sorting_method = st.selectbox(
        "Select sorting method",
        [
            "Bubble Sort", "Bubble Sort with Numpy", "Bubble Sort with Cython", "Bubble Sort with Numba",
            "Quicksort", "Quicksort with Numpy", "Quicksort with Cython", "Quicksort with Numba",
            "Merge Sort", "Merge Sort with Numpy", "Merge Sort with Cython", "Merge Sort with Numba",
            "Heap Sort", "Heap Sort with Numpy", "Heap Sort with Cython", "Heap Sort with Numba",
            "Selection Sort", "Selection Sort with Numpy", "Selection Sort with Cython", "Selection Sort with Numba",
            "Timsort", "Timsort with Numpy", "Timsort with Cython", "Timsort with Numba"
        ]
    )
    num_iterations = st.selectbox(
        "Select number of iterations for averaging runtime",
        [10, 100, 500]
    )

    if st.button("Sort DataFrame"):
        total_time = 0
        for _ in range(num_iterations):
            if sorting_method == "Bubble Sort":
                sorted_df, time_taken = bubble_sort(df.copy(), col_name, ascending)
            elif sorting_method == "Bubble Sort with Numpy":
                sorted_df, time_taken = bubble_sort_numpy(df.copy(), col_name, ascending)
            elif sorting_method == "Bubble Sort with Cython":
                sorted_df, time_taken = bubble_sort_cython_wrapper(df.copy(), col_name, ascending)
            elif sorting_method == "Bubble Sort with Numba":
                sorted_df, time_taken = bubble_sort_numba(df.copy(), col_name, ascending)
            elif sorting_method == "Quicksort":
                sorted_df, time_taken = quicksort_python(df.copy(), col_name, ascending)
            elif sorting_method == "Quicksort with Numpy":
                sorted_df, time_taken = quicksort_numpy(df.copy(), col_name, ascending)
            elif sorting_method == "Quicksort with Cython":
                sorted_df, time_taken = quicksort_cython_wrapper(df.copy(), col_name, ascending)
            elif sorting_method == "Quicksort with Numba":
                sorted_df, time_taken = quicksort_numba(df.copy(), col_name, ascending)
            elif sorting_method == "Merge Sort":
                sorted_df, time_taken = merge_sort_python(df.copy(), col_name, ascending)
            elif sorting_method == "Merge Sort with Numpy":
                sorted_df, time_taken = merge_sort_numpy(df.copy(), col_name, ascending)
            elif sorting_method == "Merge Sort with Cython":
                sorted_df, time_taken = merge_sort_cython_wrapper(df.copy(), col_name, ascending)
            elif sorting_method == "Merge Sort with Numba":
                sorted_df, time_taken = merge_sort_numba(df.copy(), col_name, ascending)
            elif sorting_method == "Heap Sort":
                sorted_df, time_taken = heap_sort_python(df.copy(), col_name, ascending)
            elif sorting_method == "Heap Sort with Numpy":
                sorted_df, time_taken = heap_sort_numpy(df.copy(), col_name, ascending)
            elif sorting_method == "Heap Sort with Cython":
                sorted_df, time_taken = heap_sort_cython_wrapper(df.copy(), col_name, ascending)
            elif sorting_method == "Heap Sort with Numba":
                sorted_df, time_taken = heap_sort_numba(df.copy(), col_name, ascending)
            elif sorting_method == "Selection Sort":
                sorted_df, time_taken = selection_sort_python(df.copy(), col_name, ascending)
            elif sorting_method == "Selection Sort with Numpy":
                sorted_df, time_taken = selection_sort_numpy(df.copy(), col_name, ascending)
            elif sorting_method == "Selection Sort with Cython":
                sorted_df, time_taken = selection_sort_cython_wrapper(df.copy(), col_name, ascending)
            elif sorting_method == "Selection Sort with Numba":
                sorted_df, time_taken = selection_sort_numba(df.copy(), col_name, ascending)
            elif sorting_method == "Timsort":
                sorted_df, time_taken = timsort_python(df.copy(), col_name, ascending)
            elif sorting_method == "Timsort with Numpy":
                sorted_df, time_taken = timsort_numpy(df.copy(), col_name, ascending)
            elif sorting_method == "Timsort with Cython":
                sorted_df, time_taken = timsort_cython_wrapper(df.copy(), col_name, ascending)
            elif sorting_method == "Timsort with Numba":
                sorted_df, time_taken = timsort_numba(df.copy(), col_name, ascending)
            total_time += time_taken

        avg_time_taken = total_time / num_iterations
        st.write("Sorted DataFrame:")
        st.write(sorted_df)
        st.write(f"Average time taken to sort over {num_iterations} iterations: {avg_time_taken:.6f} seconds")

    st.write("## Compare Sorting Methods")
    compare_method = st.selectbox(
        "Select sorting method to compare",
        [
            "Bubble Sort", "Quicksort", "Merge Sort", "Heap Sort", "Selection Sort", "Timsort"
        ]
    )

    if st.button("Compare Sorting Methods"):
        times = []
        methods = ["Python", "Numpy", "Cython", "Numba"]

        if compare_method == "Bubble Sort":
            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = bubble_sort(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = bubble_sort_numpy(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = bubble_sort_cython_wrapper(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = bubble_sort_numba(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

        elif compare_method == "Quicksort":
            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = quicksort_python(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = quicksort_numpy(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = quicksort_cython_wrapper(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = quicksort_numba(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

        elif compare_method == "Merge Sort":
            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = merge_sort_python(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = merge_sort_numpy(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = merge_sort_cython_wrapper(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = merge_sort_numba(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

        elif compare_method == "Heap Sort":
            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = heap_sort_python(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = heap_sort_numpy(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = heap_sort_cython_wrapper(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = heap_sort_numba(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

        elif compare_method == "Selection Sort":
            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = selection_sort_python(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = selection_sort_numpy(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = selection_sort_cython_wrapper(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = selection_sort_numba(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

        elif compare_method == "Timsort":
            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = timsort_python(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = timsort_numpy(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = timsort_cython_wrapper(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

            total_time = 0
            for _ in range(num_iterations):
                _, time_taken = timsort_numba(df.copy(), col_name, ascending)
                total_time += time_taken
            times.append(total_time / num_iterations)

        fig, ax = plt.subplots()
        ax.bar(methods, times)
        ax.set_xlabel("Method")
        ax.set_ylabel("Average Time (seconds)")
        ax.set_title(f"Comparison of {compare_method} Methods over {num_iterations} iterations")
        st.pyplot(fig)