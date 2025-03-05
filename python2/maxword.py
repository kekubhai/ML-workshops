with open ("length.txt", "w+")as file:
    content=input("Enter your content")
    file.write(content)
    file.seek(0)
    words=file.read().split()
    longest=max(words, key=len)
    print(f"Longest word is:", {longest})