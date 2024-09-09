using CSV
using DataFrames

name = "HFSMB"
# Step 1: Load the CSV file
data = CSV.read("$name.csv", DataFrame, header=false)

# Function to categorize rows based on their index in the DataFrame
function categorize_row_index(i)
    bin_size = 79  # Each range size
    return ceil(Int, i / bin_size)  # Determine the bin/category for each row
end

# Apply the function to each row index and add as a new column
data.category = [categorize_row_index(i) for i in 1:nrow(data)]

# Aggregate the sum of the third column by the new category column
# Assume the third column is named :Column3, replace with the actual name if different
aggregated_data = combine(groupby(data, :category), :Column3 => sum => :Column3Sum)

# Save the aggregated data to a new CSV file
CSV.write("agg_$(name).csv", aggregated_data)