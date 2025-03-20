import pandas as pd
import time

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

# Optimized Numeric Filter
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

# Non-Optimized Numeric Filter (Slower)
def slow_filter_numeric_column(data, column, condition, value):
    """Intentionally slower filtering for numeric columns."""
    start_time = time.time()
    result = []
    for index, row in data.iterrows():
        if condition == 'greater_than' and row[column] > value:
            result.append(row)
        elif condition == 'less_than' and row[column] < value:
            result.append(row)
        elif condition == 'equal_to' and row[column] == value:
            result.append(row)
        elif condition == 'not_equal_to' and row[column] != value:
            result.append(row)
    end_time = time.time()
    print(f"Slow method took {end_time - start_time:.4f} seconds")
    return pd.DataFrame(result)

# Optimized String Filter
def filter_string_column(data, column, condition, value):
    """Apply optimal filtering for string columns."""
    if condition == 'contains':
        return data[data[column].str.contains(value, na=False)]
    elif condition == 'startswith':
        return data[data[column].str.startswith(value, na=False)]
    elif condition == 'endswith':
        return data[data[column].str.endswith(value, na=False)]
    elif condition == 'in':
        return data[data[column].isin(value)]
    else:
        raise ValueError(f"Condition '{condition}' not recognized for string columns.")

# Non-Optimized String Filter (Slower)
def slow_filter_string_column(data, column, condition, value):
    """Intentionally slower filtering for string columns."""
    start_time = time.time()
    result = []
    for index, row in data.iterrows():
        if condition == 'contains' and value in row[column]:
            result.append(row)
        elif condition == 'startswith' and row[column].startswith(value):
            result.append(row)
        elif condition == 'endswith' and row[column].endswith(value):
            result.append(row)
        elif condition == 'in' and row[column] in value:
            result.append(row)
    end_time = time.time()
    print(f"Slow method took {end_time - start_time:.4f} seconds")
    return pd.DataFrame(result)

# Optimized Categorical Filter
def filter_categorical_column(data, column, condition, value):
    """Apply optimal filtering for categorical columns."""
    data[column] = data[column].astype(str)
    
    if condition == 'in':
        return data[data[column].isin(value)]
    elif condition == 'startswith':
        return data[data[column].str.startswith(value, na=False)]
    elif condition == 'contains':
        return data[data[column].str.contains(value, na=False)]
    elif condition == 'endswith':
        return data[data[column].str.endswith(value, na=False)]
    else:
        raise ValueError(f"Condition '{condition}' not recognized for categorical columns.")

# Non-Optimized Categorical Filter (Slower)
def slow_filter_categorical_column(data, column, condition, value):
    """Intentionally slower filtering for categorical columns."""
    data[column] = data[column].astype(str)
    start_time = time.time()
    result = []
    for index, row in data.iterrows():
        if condition == 'contains' and value in row[column]:
            result.append(row)
        elif condition == 'startswith' and row[column].startswith(value):
            result.append(row)
        elif condition == 'endswith' and row[column].endswith(value):
            result.append(row)
        elif condition == 'in' and row[column] in value:
            result.append(row)
    end_time = time.time()
    print(f"Slow method took {end_time - start_time:.4f} seconds")
    return pd.DataFrame(result)

# Main function to apply the appropriate filter
def filter_column(data, column, condition, value, slow=False):
    """General function to filter a column based on its type and the selected condition."""
    column_type = identify_column_type(data, column)
    
    if slow:
        # Apply slow filtering method based on the column type
        if column_type == 'numeric':
            return slow_filter_numeric_column(data, column, condition, value)
        elif column_type == 'string':
            return slow_filter_string_column(data, column, condition, value)
        elif column_type == 'categorical':
            return slow_filter_categorical_column(data, column, condition, value)
    else:
        # Apply optimized filtering
        if column_type == 'numeric':
            return filter_numeric_column(data, column, condition, value)
        elif column_type == 'string':
            return filter_string_column(data, column, condition, value)
        elif column_type == 'categorical':
            return filter_categorical_column(data, column, condition, value)
    raise ValueError(f"Unsupported column type '{column_type}' for filtering.")

