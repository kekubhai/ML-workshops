import csv

count = 0

# Open the CSV file
with open('student.csv', 'r') as file:
    reader = csv.reader(file)
    # Skip the header row
    next(reader)
    
    # Iterate through each row
    for row in reader:
        # Check if the score is more than 80
        if int(row[2]) > 80:
            count += 1

# Display the count
print(f"Number of records where score is more than 80: {count}")
