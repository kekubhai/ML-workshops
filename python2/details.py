import csv

header = ["Country", "Played", "Won", "Lost", "Tied", "No Result"]
n1 = ["England", "746", "375", "334", "9", "28"]
n2 = ["Australia", "932", "566", "323", "9", "34"]
n3 = ["India", "97", "513", "417", "11", "46"]

new_rows = [
    ['New Zealad', 500, 250, 200, 10, 40],
    ['Sout Africa', 600, 300, 250, 20, 30]
]

data = [
    {'Country': 'Pakistan', 'Played': 400, 'Won': 200, 'Lost': 180, 'Tied': 10, 'No Result': 10},
    {'Country': 'Sri Lanka', 'Played': 450, 'Won': 220, 'Lost': 200, 'Tied': 15, 'No Result': 15}
]

# Writing data into CSV (Rows + New Rows)
with open('details.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerow(n1)
    writer.writerow(n2)
    writer.writerow(n3)
    writer.writerows(new_rows)
    
print("CSV file created successfully")

# Appending Dictionary Data to CSV
with open('details.csv', mode='a', newline='') as file:
    fieldnames = ["Country", "Played", "Won", "Lost", "Tied", "No Result"]
    csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    # Writing dictionary data without writing headers again
    csv_writer.writerows(data)

print("Dictionary data appended successfully")
 
    