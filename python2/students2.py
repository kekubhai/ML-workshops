import csv

students = []
for i in range(2):
    rollno = input("Enter Roll No: ")
    name = input("Enter Name: ")
    class_ = input("Enter Class: ")
    students.append([rollno, name, class_])


with open('Student.csv', mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Rollno', 'Name', 'Class'])  # Write header
    csv_writer.writerows(students)