with open("mylife.txt", "w+") as file1, open("newlife.txt", "w+") as file2:
    lines = ["this is line 1", " this is line 2", "This is line 3"]
    newlines = ["this is new line 1", " this is new line 2", "This is new line 3"]

    for line in lines:
        file1.write(line + "\n")
    file1.seek(0)  # Reset cursor to the beginning
    old = file1.read()

    for newline in newlines:
        file2.write(newline + "\n")
    file2.seek(0)  # Reset cursor to the beginning
    new = file2.read()

    if old == new:
        print("Files are same")
    else:
        print("Files are different")
