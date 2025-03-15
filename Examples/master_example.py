from filtering_example import load_csv, filter_column
import pandas as pd

def main():
    # Provide CSV path example and a loading message
    file_path = input("Enter the path to your CSV file (e.g., /path/to/your/data.csv): ")
    print("Please give it a moment. This might take a while to load depending on the file size and your machine's specs.")
    
    # Load the CSV file
    data = load_csv(file_path)

    # Keep asking for action until the user chooses 'filter'
    while True:
        # Ask what the user wants to do: filter, sort, or group
        action = input("What would you like to do? (filter, sort, group): ").lower()

        # Handle actions
        if action == 'sort':
            print("Sorry, still working on sorting.")
        elif action == 'group':
            print("Sorry, still working on grouping.")
        elif action == 'filter':
            break  # Exit the loop if the user selects 'filter'
        else:
            print("Invalid choice. Please select 'filter', 'sort', or 'group'.")

    # If the user wants to filter, ask for a column
    print("Available columns:", data.columns)
    column = input("Enter the column you want to filter: ")

    # Get the column's data type and inform the user
    column_type = data[column].dtype
    print(f"\nThe column '{column}' is of type {column_type}.")

    # Show a preview of the values in the selected column
    print(f"\nHere are some example values from the '{column}' column:")
    print(data[column].head())  # Display the first 5 values for the user

    # Explain what type of value the user should enter based on the column's datatype
    if column_type in ['int64', 'float64']:
        print("Since this is a numeric column, please enter a numeric value (e.g., a number or float).")
    elif column_type == 'object':
        print("Since this is a string or categorical column, please enter a string value (e.g., text).")
    elif column_type == 'datetime64[ns]':
        print("Since this is a datetime column, please enter a date (e.g., 'YYYY-MM-DD').")
    else:
        print("Unknown column type. Please enter a valid value according to the column's data type.")

    # List of available filter conditions with explanations
    print("\nChoose one of the following conditions for filtering:")
    print("1. greater_than")
    print("2. less_than")
    print("3. equal_to")
    print("4. not_equal_to")
    print("5. contains (for string columns)")
    print("6. startswith (for string columns)")
    print("7. endswith (for string columns)")
    print("8. in (for categorical columns)")

    # Ask the user for the condition they want to apply
    condition = input("\nEnter the condition (e.g., greater_than, less_than, etc.): ").lower()

    # Get the value for the condition from the user
    value = input("Enter the value for the condition: ")

    # Convert the value to the correct type based on the column's data type and condition
    if column_type in ['int64', 'float64']:
        # For numeric conditions (greater_than, less_than, etc.), convert the value to float
        value = float(value)
        if condition == 'between':  # If the condition is 'between', expect a tuple of values
            value = tuple(map(float, value.split(',')))  # Convert a comma-separated string to a tuple
    elif column_type == 'object':
        # For string-based conditions (contains, startswith, endswith), keep the value as a string
        value = str(value)
    elif column_type == 'datetime64[ns]':
        # For datetime columns, convert the value to a date
        value = pd.to_datetime(value)
    
    # Apply the filter
    filtered_data = filter_column(data, column, condition, value)

    # Display the filtered data
    print("\nFiltered Data:")
    print(filtered_data)

if __name__ == "__main__":
    main()
