import pandas as pd

# Load the CSV file
csv_file_path = "C:\\Users\\Khang Nguyen\\Documents\\GitHub\\Chinese_Postman_Problem\\experiments\\heuristic_results.csv"
data = pd.read_csv(csv_file_path)

# Function to print average cost and time for specified conditions
def print_averages(condition_name, condition):
    filtered_data = data[condition]
    if not filtered_data.empty:
        avg_cost = filtered_data['Cost'].mean()
        avg_time = filtered_data['Time (s)'].mean()
        print(f"Average cost and time for {condition_name}:")
        print(f"  Average Cost: {avg_cost:.2f}, Average Time: {avg_time:.2f} seconds")
    else:
        print(f"No data found for {condition_name}.")

# 1. Average for test cases starting with "C_"
print_averages("C_", data['Name'].str.startswith("C_"))

# 2. Average for test cases starting with "E_"
print_averages("E_", data['Name'].str.startswith("E_"))

# 3. Average for test cases starting with "H_"
print_averages("H_", data['Name'].str.startswith("H_"))

# 4. Average for test cases starting with "small_" and number of edges = 8
print_averages("small_ with edges = 8", data['Name'].str.startswith("small_") & (data['Number of edges'] == 8))

# 5. Average for test cases starting with "small_" and number of edges = 9
print_averages("small_ with edges = 9", data['Name'].str.startswith("small_") & (data['Number of edges'] == 9))

# 6. Average for test cases starting with "small_" and number of edges = 10
print_averages("small_ with edges = 10", data['Name'].str.startswith("small_") & (data['Number of edges'] == 10))

# 7. Average for test cases starting with "large_" and number of edges in {16, 20, 27}
print_averages("large_ with edges in {16, 20, 27}", data['Name'].str.startswith("large_") & data['Number of edges'].isin([16, 20, 27]))

# 8. Average for test cases starting with "large_" and number of edges in {53, 75}
print_averages("large_ with edges in {53, 75}", data['Name'].str.startswith("large_") & data['Number of edges'].isin([53, 75]))

# 9. Average for test cases starting with "large_" and number of edges in {110, 162, 232}
print_averages("large_ with edges in {110, 162, 232}", data['Name'].str.startswith("large_") & data['Number of edges'].isin([110, 162, 232]))
