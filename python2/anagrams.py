file = open("words.txt", "w+")
words = file.read().split()  # Reading file and splitting words
file.close()

anagram_count = 0
checked = []  # To avoid double counting

for word1 in words:
    for word2 in words:
        if word1 != word2 and sorted(word1) == sorted(word2) and (word1, word2) not in checked and (word2, word1) not in checked:
            anagram_count += 1
            checked.append((word1, word2))

print("Number of Anagram Pairs:", anagram_count)
