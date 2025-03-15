import pandas as pd

def load_csv(file_path):
    """Load CSV into pandas DataFrame."""
    return pd.read_csv(file_path)

def identify_column_type(data, column):
    """Identify the data type of the selected column."""
    if pd.api.types.is_numeric_dtype(data[column]):
        return 'numeric'
    elif pd.api.types.is_string_dtype(data[column]):
        return 'string'
    elif pd.api.types.is_categorical_dtype(data[column]) or pd.api.types.is_object_dtype(data[column]):
        return 'categorical'
    else:
        raise ValueError(f"Column '{column}' has an unsupported data type.")

def filter_numeric_column(data, column, condition, value):
    """Apply optimal filtering for numeric columns."""
    if condition == 'greater_than':
        return data[data[column] > value]
    elif condition == 'less_than':
        return data[data[column] < value]
    elif condition == 'equal_to':
        return data[data[column] == value]
    elif condition == 'not_equal_to':
        return data[data[column] != value]
    else:
        raise ValueError(f"Condition '{condition}' not recognized for numeric columns.")

def filter_string_column(data, column, condition, value):
    """Apply optimal filtering for string columns."""
    if condition == 'contains':
        return data[data[column].str.contains(value, na=False)]
    elif condition == 'startswith':
        return data[data[column].str.startswith(value, na=False)]
    elif condition == 'endswith':
        return data[data[column].str.endswith(value, na=False)]
    else:
        raise ValueError(f"Condition '{condition}' not recognized for string columns.")

def filter_categorical_column(data, column, condition, value):
    """Apply optimal filtering for categorical columns."""
    if condition == 'in':
        return data[data[column].isin(value)]
    else:
        raise ValueError(f"Condition '{condition}' not recognized for categorical columns.")

def filter_column(data, column, condition, value):
    """General function to filter a column based on its type and the selected condition."""
    column_type = identify_column_type(data, column)
    
    if column_type == 'numeric':
        return filter_numeric_column(data, column, condition, value)
    elif column_type == 'string':
        return filter_string_column(data, column, condition, value)
    elif column_type == 'categorical':
        return filter_categorical_column(data, column, condition, value)
    else:
        raise ValueError(f"Unsupported column type '{column_type}' for filtering.")

# Main function to interact with user and apply filter
def main():
    # Load CSV file
    file_path = input("Enter the path to your CSV file: ")
    data = load_csv(file_path)

    # Display available columns
    print("Available columns:", data.columns)

    # Get column, condition, and value for filtering
    column = input("Enter the column you want to filter: ")
    condition = input("Enter the condition (greater_than, less_than, equal_to, contains, etc.): ")
    value = input("Enter the value for the condition: ")

    # Convert value to appropriate type
    if condition in ['greater_than', 'less_than', 'equal_to', 'not_equal_to']:
        value = float(value)
    elif condition in ['contains', 'startswith', 'endswith']:
        value = str(value)
    elif condition == 'in':
        value = value.split(',')  # Convert a comma-separated string to a list of values

    # Apply the filtering
    filtered_data = filter_column(data, column, condition, value)

    # Show the filtered data
    print("Filtered Data:")
    print(filtered_data)

if __name__ == "__main__":
    main()
