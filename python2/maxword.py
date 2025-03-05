with open("data.txt", "r") as file:
    lines = file.readlines()  # Read all lines into a list

longest_line = max(lines, key=len)  # Find the longest line using max()

print("Longest Line:", longest_line)
print("Length of Longest Line:", len(longest_line))
