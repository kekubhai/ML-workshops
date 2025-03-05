with open('data.txt', 'r') as file:
    words = file.read().split()
    fre1 = {word: words.count(word) for word in words}
    print("Word Frequency:", fre1)
    print("Unique Words:", set(words))
