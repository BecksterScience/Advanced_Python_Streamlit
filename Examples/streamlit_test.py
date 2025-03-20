import streamlit as st
import pandas as pd
from filtering_example import load_csv, filter_column  # Assuming filter_column is the function in your filtering.py
import time
import matplotlib.pyplot as plt

def main():
    # Streamlit UI components
    st.title("Analyzing Most Efficient Search/Sort/Filter Algorithms Based Off Data")

    # Step 1: Ask for file upload
    file = st.file_uploader("Upload a CSV file", type=["csv"])

    if file is not None:
        # Load the CSV data
        data = load_csv(file)
        
        # Step 2: Let the user select a column to filter
        column = st.selectbox("Select column to filter", data.columns)
        
        # Get the column's data type
        column_type = data[column].dtype
        st.write(f"The column '{column}' is of type {column_type}.")

        # Step 3: Show preview of values (head and tail)
        st.write(f"Here are the first 5 values from the '{column}' column:")
        st.write(data[column].head())  # Display the first 5 values
        st.write(f"Here are the last 5 values from the '{column}' column:")
        st.write(data[column].tail())  # Display the last 5 values

        # Step 4: Display Min and Max values if numeric
        if column_type in ['int64', 'float64']:
            st.write(f"The minimum value in '{column}' is: {data[column].min()}")
            st.write(f"The maximum value in '{column}' is: {data[column].max()}")

        # Step 5: Algorithm recommendation (based on column type)
        st.write(f"Based on this column's type ({column_type}), the most optimal filtering algorithm is (to be added):")
        st.write("**Best Filtering Algorithm**: [Display filtering algorithm logic here]")

        # Step 6: Ask how the user would like to filter based on the data type
        if column_type in ['int64', 'float64']:  # If numeric column, show numeric filter options
            filter_operation = st.selectbox(
                "How would you like to filter this numeric column?",
                ["Greater than", "Less than", "Equal to", "Between"]
            )
            
            # Convert the filter operation to lowercase to match the function's expected format
            filter_operation = filter_operation.lower().replace(" ", "_")  # Convert to 'greater_than', etc.

            # Ask for the value(s) based on the filter operation
            if filter_operation == "greater_than":
                filter_value = st.number_input(f"Enter the value for '{filter_operation}'", min_value=data[column].min())
            elif filter_operation == "less_than":
                filter_value = st.number_input(f"Enter the value for '{filter_operation}'", max_value=data[column].max())
            elif filter_operation == "equal_to":
                filter_value = st.number_input(f"Enter the value for '{filter_operation}'")
            elif filter_operation == "between":
                filter_value_min = st.number_input(f"Enter the minimum value for '{filter_operation}'", min_value=data[column].min())
                filter_value_max = st.number_input(f"Enter the maximum value for '{filter_operation}'", max_value=data[column].max())
        
        elif column_type == 'object' or column_type == 'category':  # For string/categorical columns
            filter_operation = st.selectbox(
                "How would you like to filter this categorical/string column?",
                ["Contains", "Starts with", "Ends with", "In"]
            )
            
            if filter_operation in ["Contains", "Starts with", "Ends with"]:
                filter_value = st.text_input(f"Enter the value for '{filter_operation}'. Don't include '' (quotes).")
            elif filter_operation == "In":
                filter_value = st.text_input(f"Enter a comma-separated list of values to include in filter")
                filter_value = filter_value.split(",")  # Convert input string to list
        
        # Convert filter_operation to lowercase to match the function's expected format
        filter_operation = filter_operation.lower()

        # Ensure that the filter value is converted correctly for numeric types
        if filter_operation in ["greater_than", "less_than", "equal_to"]:
            try:
                filter_value = float(filter_value)  # Ensure the numeric input is converted to a float
            except ValueError:
                st.error("Please enter a valid numeric value for the filter.")
                return  # Exit if invalid numeric input
        
        # Apply filtering based on user's choice
        if filter_operation in ["greater_than", "less_than", "equal_to"]:
            if filter_value:
                filtered_data = filter_column(data, column, filter_operation, filter_value)
                st.write(f"Filtered data using {filter_operation} for {column}:")
                st.write(filtered_data)
        elif filter_operation == "between":
            if filter_value_min and filter_value_max:
                filtered_data = filter_column(data, column, filter_operation, filter_value_min, filter_value_max)
                st.write(f"Filtered data between {filter_value_min} and {filter_value_max} for {column}:")
                st.write(filtered_data)
        elif filter_operation in ["contains", "starts with", "ends with"]:
            if filter_value:
                filtered_data = filter_column(data, column, filter_operation, filter_value)
                st.write(f"Filtered data using {filter_operation} for {column}:")
                st.write(filtered_data)
        elif filter_operation == "in":
            if filter_value:
                filtered_data = filter_column(data, column, filter_operation, filter_value)
                st.write(f"Filtered data using {filter_operation} for {column}:")
                st.write(filtered_data)

        # Display the most efficient way found (from filter.py logic)
        st.write(f"For this operation and data type, the most efficient filtering method found is:")
        st.write("**Efficient Filtering Algorithm**: [Display filtering algorithm logic here]")

        # Step 7: Ask if the user wants to see performance metrics for the method just employed
        performance_metrics = st.radio(
            "Would you like to see the performance metrics for the filtering method just employed?",
            ('Yes', 'No')
        )

        if performance_metrics == 'Yes':
            # Measure time for performance of the optimized method
            start_time = time.time()
            filtered_data_optimized = filter_column(data, column, filter_operation, filter_value)
            optimized_time = time.time() - start_time  # Calculate the time taken for optimized filtering

            # Display the performance metrics for the optimized method
            st.write(f"Filtered data using {filter_operation} for {column} (Optimized method):")
            st.write(filtered_data_optimized)
            st.write(f"Filtering time for optimized method: {optimized_time:.4f} seconds")

            # Step 7: Create dynamic subset sizes based on the total number of rows in the dataset
            total_rows = len(data)
            subset_percentages = [i for i in range(2, 101, 2)]  # 2%, 4%, 6%, ..., 100%

            # Create subsets and measure performance times for optimized and slower methods
            optimized_times = []
            slow_times = []
            subset_sizes = []

            # Add progress bar to show progress
            progress_bar = st.progress(0)

            for idx, percent in enumerate(subset_percentages):
                subset_size = int((percent / 100) * total_rows)
                if subset_size == 0:
                    continue  # Skip if the subset size is zero

                # Subset data and measure time for optimized method
                subset_data = data.sample(n=subset_size)
                start_time = time.time()
                filter_column(subset_data, column, filter_operation, filter_value)
                optimized_times.append(time.time() - start_time)

                # Measure time for slower method
                start_time = time.time()
                filter_column(subset_data, column, filter_operation, filter_value, slow=True)
                slow_times.append(time.time() - start_time)

                subset_sizes.append(subset_size)

                # Update progress bar
                progress_bar.progress((idx + 1) / len(subset_percentages))

            # Plot the performance comparison
            fig, ax = plt.subplots()
            ax.plot(subset_sizes, optimized_times, label='Optimized Method', marker='o')
            ax.plot(subset_sizes, slow_times, label='Slower Method', marker='x')
            ax.set_xlabel("Data Size (Subset Size in Number of Rows)")
            ax.set_ylabel("Time (seconds)")
            ax.set_title(f"Performance Comparison for {filter_operation} on {column}")
            ax.legend()
            st.pyplot(fig)

        # Step 8: Ask if the user wants to see the optimization with parallel processing
        optimization_choice = st.radio(
            "Do you want to see how the best method can be optimized when we apply parallel processing?",
            ('Yes', 'No')
        )

        if optimization_choice == 'Yes':
            # Measure time for performance of the parallel processing method
            start_time = time.time()
            filtered_data_parallel = filter_column(data, column, filter_operation, filter_value, parallel=True)
            parallel_time = time.time() - start_time  # Calculate the time taken for parallel processing

            # Display the performance metrics for the parallel method
            st.write(f"Filtered data using {filter_operation} for {column} (Parallel method):")
            st.write(filtered_data_parallel)
            st.write(f"Filtering time for parallel method: {parallel_time:.4f} seconds")

            # Step 8: Plot the comparison between optimized and parallel methods
            fig, ax = plt.subplots()
            ax.bar(['Optimized Method', 'Parallel Method'], [optimized_time, parallel_time])
            ax.set_ylabel("Time (seconds)")
            ax.set_title(f"Comparison between Optimized and Parallel Processing for {filter_operation} on {column}")
            st.pyplot(fig)

        # If the user chooses "No", continue to the next step
        if optimization_choice == 'No':
            st.write("Great! Let's move on to the next step.")

if __name__ == "__main__":
    main()
