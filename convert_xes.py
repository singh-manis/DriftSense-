import pm4py
import pandas as pd
import os

def convert_xes_to_numeric_csv(input_file, output_file):
    print(f"Loading {input_file}... (This might take a moment)")
    
    # 1. Read the XES file
    log = pm4py.read_xes(input_file)
    
    # 2. Convert to Dataframe
    df = pm4py.convert_to_dataframe(log)
    
    print("Converting process data to numeric features...")

    # 3. Create a 'Bag-of-Activities' (Count activities per case)
    # This turns text activities like "Process Payment" into numbers (e.g., 1, 0)
    # Rows = Cases, Columns = Activity Names
    case_id_col = "case:concept:name"
    activity_col = "concept:name"
    
    # Group by Case ID and count the activities
    numeric_df = pd.crosstab(df[case_id_col], df[activity_col])
    
    # 4. Save to CSV
    numeric_df.to_csv(output_file, index=False)
    print(f"Success! Converted data saved to: {output_file}")
    print(f"Columns (Features): {list(numeric_df.columns)}")

if __name__ == "__main__":
    # CHANGE THIS to your actual file name
    INPUT_FILENAME = "bpi_log.csv.xes" 
    OUTPUT_FILENAME = "converted_data.csv"
    
    if os.path.exists(INPUT_FILENAME):
        convert_xes_to_numeric_csv(INPUT_FILENAME, OUTPUT_FILENAME)
    else:
        print(f"Error: Could not find '{INPUT_FILENAME}'. Please check the file name.")