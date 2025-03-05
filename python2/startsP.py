with open('data.txt','r')as file:
    words=file.readlines()
    starts_with_P=[word for word in words if word.startswith('P')]
    print("Words starting with 'P':",starts_with_P)