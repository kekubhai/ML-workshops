with open("data.txt", "r") as file:
    content = file.read()  # Read file content
    
words = content.split()  # Split content into words
unique_words = []  # List to store unique words

for word in words:
    if word not in unique_words:
        unique_words.append(word)

new_content = " ".join(unique_words)  # Join unique words back into text

with open("data.txt", "w") as file:
    file.write(new_content)  # Write the updated content

print("Duplicate words deleted successfully!")
