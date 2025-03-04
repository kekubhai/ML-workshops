import csv


credentials = []
for i in range(3):
    username = input("Enter username: ")
    password = input("Enter password: ")
    credentials.append([username, password])


with open('login_credentials.csv', mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Username', 'Password'])  # Write header
    csv_writer.writerows(credentials)


with open('login_credentials.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        print(row)