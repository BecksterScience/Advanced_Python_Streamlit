import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from filtering_example import filter_column  # Import the filter_column function from your existing filtering file

def filter_parallel(data, column, condition, value, chunk_size=1000, slow=False):
    """Apply parallel processing to the filtering task."""
    # Split the data into chunks
    num_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size != 0 else 0)
    chunks = [data.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    
    # Use ProcessPoolExecutor to apply the filter in parallel
    with ProcessPoolExecutor() as executor:
        futures = []
        
        for chunk in chunks:
            futures.append(executor.submit(filter_column, chunk, column, condition, value, slow))
        
        # Wait for all futures to complete and combine the results
        results = [future.result() for future in futures]
        
    # Combine all the filtered chunks back together
    return pd.concat(results, ignore_index=True)