from collections import defaultdict

filename = input("Enter the filename: ")

with open(filename, 'w') as file:
    content = input("Enter the content for the file: ")
    file.write(content)

def count_anagrams(filename):
    with open(filename, 'r') as file:
        words = file.read().split()
    
    word_dict = defaultdict(list)
    for word in words:
        sorted_word = ''.join(sorted(word))
        word_dict[sorted_word].append(word)
    
    anagram_count = sum(len(values) > 1 for values in word_dict.values())
    return anagram_count

# Example usage:
print(f"Number of anagram groups: {count_anagrams(filename)}")
