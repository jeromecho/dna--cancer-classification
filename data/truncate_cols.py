import pandas as pd

# Load the CSV file while restricting to the first 37 columns
df = pd.read_csv("partial-data.csv", usecols=range(37))

# Save the processed CSV file
df.to_csv("processed-data.csv", index=False)

print("Preprocessing complete: restricted to the first 37 columns.")
