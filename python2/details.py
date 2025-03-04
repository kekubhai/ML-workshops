import csv
header={"Country","Played", "Won", "Lost","Tied", "No Result"}
n1={"England", "746", "375","334","9","28"}
n2={"Australia", "932", "566","323","9","34"}
n3={"India", "987", "513","417", "11","46"}
with open ('details.csv', 'w') as file :
    writer=csv.writer(file)
    