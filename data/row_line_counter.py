import csv

with open("./content/KICH.csv", newline="") as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Read header
    print(f"Number of columns: {len(header)}")  # Check column count in header
    
    for row in reader:
        print(f"Row has {len(row)} elements")