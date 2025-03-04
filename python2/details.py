import csv
header = ["Country", "Played", "Won", "Lost", "Tied", "No Result"]
n1 = ["England", "746", "375", "334", "9", "28"]
n2 = ["Australia", "932", "566", "323", "9", "34"]
n3 = ["India", "987", "513", "417", "11", "46"]

with open('details.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Writing Header
    writer.writerow(n1)     # Writing Row 1
    writer.writerow(n2)     # Writing Row 2
    writer.writerow(n3)     # Writing Row 3

print("CSV file created successfully")
