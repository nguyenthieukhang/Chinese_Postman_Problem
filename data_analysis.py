import pandas as pd
import sys
from tabulate import tabulate

# Check for command-line argument
if len(sys.argv) < 2:
    print("Usage: python data_analysis.py <csv_file_path>")
    sys.exit(1)

# Get the CSV file path from the command line
csv_file_path = sys.argv[1]

# Load the CSV file
try:
    data = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: File not found at {csv_file_path}")
    sys.exit(1)

# Function to calculate average cost and time for specified conditions
def calculate_averages(condition_name, condition):
    filtered_data = data[condition]
    if not filtered_data.empty:
        avg_cost = filtered_data['Cost'].mean()
        avg_time = filtered_data['Time (s)'].mean()
        return condition_name, avg_cost, avg_time
    else:
        return condition_name, None, None

# Define conditions
conditions = [
    ("C_", data['Name'].str.startswith("C_")),
    ("E_", data['Name'].str.startswith("E_")),
    ("H_", data['Name'].str.startswith("H_")),
    ("small_ with edges = 8", data['Name'].str.startswith("small_") & (data['Number of edges'] == 8)),
    ("small_ with edges = 9", data['Name'].str.startswith("small_") & (data['Number of edges'] == 9)),
    ("small_ with edges = 10", data['Name'].str.startswith("small_") & (data['Number of edges'] == 10)),
    ("large_ with edges in {16, 20, 27}", data['Name'].str.startswith("large_") & data['Number of edges'].isin([16, 20, 27])),
    ("large_ with edges in {53, 75}", data['Name'].str.startswith("large_") & data['Number of edges'].isin([53, 75])),
    ("large_ with edges in {110, 162, 232}", data['Name'].str.startswith("large_") & data['Number of edges'].isin([110, 162, 232]))
]

# Compute results
results = [calculate_averages(name, condition) for name, condition in conditions]

# Create a DataFrame for better formatting
results_df = pd.DataFrame(results, columns=["Condition", "Average Cost", "Average Time (s)"])

# Output results as a table
print(tabulate(results_df, headers="keys", tablefmt="pretty"))
