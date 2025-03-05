


with open("mylife.txt", "w") as file :
    lines=["this is line 1", " this is line 2", "This is line 3"]
    for line in lines:
        file.write(line+ "\n")
        