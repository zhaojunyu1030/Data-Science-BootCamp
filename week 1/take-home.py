# 1. Write a function count_vowels(word) that takes a word as an argument and returns the number of vowels in the word
def count_vowels(word):
    count = 0
    for char in word:
        if char == 'a' or char == 'e' or char == 'i' or char == 'o' or char == 'u':
            count += 1
    return count

print(count_vowels("take-home assignment"))


# 2. Iterate through the following list of animals and print each one in all caps.
animals = ['tiger', 'elephant', 'monkey', 'zebra', 'panther']
for animal in animals:
    print(animal.upper())


# 3. Write a program that iterates from 1 to 20, printing each number and whether it's odd or even.
for i in range(1, 21):
    print(i, end='\t')
    if i % 2 == 1:
        print('odd')
    else:
        print('even')


# 4. Write a function sum_of_integers(a, b) that takes two integers as input from the user and returns their sum.
def sum_of_integers(a, b):
    return a+b

num1 = int(input("Enter value of a: "))
num2 = int(input("Enter value of b: "))
print(sum_of_integers(num1, num2))
