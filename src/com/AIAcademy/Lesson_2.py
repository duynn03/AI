import math

# Question 1:
# Write a program which will find all such numbers which are divisible by 7 but are not a multiple
# of 5 between 2000 and 3200 (both included). The numbers obtained should be printed in a
# comma-separated sequence on a single line.
# Hints: Consider use range(#begin, #end) method, join() method.

result = []
for i in range(2000, 3201):
    if (i % 7 == 0) and (i % 5 != 0):
        result.append(str(i))
print(','.join(result))

# Question 2:
# Write a program which can compute the factorial of a given numbers. The results should be
# printed in a comma-separated sequence on a single line.
# Example:
# Suppose the following input is supplied to the program: 8
# Then, the output should be: 40320
# Hint: use input() method

# C1:
factorial = 1
inputNumber = int(input("tell me a number: "))
for i in range(1, inputNumber + 1):
    factorial *= i
print(factorial)


# C2:
def fact(x):
    if x == 0:
        return 1
    return x * fact(x - 1)


x = int(input())
print(fact(x))

# Question 3:
# With a given integral number n, write a program to generate a dictionary that contains (i, i*i)
# such that is an integral number between 1 and n (both included). and then the program should
# print the dictionary.
# Example:
# Suppose the following input is supplied to the program: 8
# Then, the output should be: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64}
# Hints: use input(), dict() method

dict = {}
inputNumber = int(input("tell me a number of dictionary: "))
for i in range(1, inputNumber + 1):
    dict[i] = i * i
print(dict)

# Question 4
# Write a program which accepts a sequence of comma-separated numbers from console and
# generate a list and a tuple which contains every number.
# Example:
# Suppose the following input is supplied to the program: 34,67,55,33,12,98
# Then, the output should be:
# ['34', '67', '55', '33', '12', '98']
# ('34', '67', '55', '33', '12', '98')
# Hints: use input(), tuple() method can convert list to tuple

inputList = input("tell me a list of number: ").split(",")
print("List: " + str(inputList))
print("Tuple: " + str(tuple(inputList)))


# Question 5
# Define a class which has at least two methods:
# getString: to get a string from console input
# printString: to print the string in upper case.
# Also please include simple program to test the class methods.
# Hints: Use __init__ method to construct some parameters
class InputOutString(object):
    def __init__(self):
        self.s = ""

    def getString(self):
        self.s = input("Enter input: ")

    def printString(self):
        print("Output has Upper: " + self.s.upper())


obj = InputOutString()
obj.getString()
obj.printString()

# Question 6
# Write a program that calculates and prints the value according to the given formula:
# Q = Square root of [(2 * C * D)/H]
# Following are the fixed values of C and H: C is 50. H is 30.
# D is the variable whose values should be input to your program in a comma-separated
# sequence.
# Example:
# Let us assume the following comma separated input sequence is given to the program:
# 100,150,180
# The output of the program should be: 18,22,24
# Hints: Use spit() method. If the output received is in decimal form, it should be rounded off to its
# nearest value (for example, if the output received is 26.0, it should be printed as 26)
C = 50
H = 30
result = []
inputs = [x for x in input("Enter list number: ").split(",")]
for d in inputs:
    result.append(str(int(round(math.sqrt((2 * C * float(d)) / H)))))
print(",".join(result))


# Question 7
# Write a program which takes 2 digits, X,Y as input and generates a 2-dimensional array. The
# element value in the i-th row and j-th column of the array should be i*j.
# Note: i=0,1.., X-1; j=0,1,ยก-Y-1.
# Example
# Suppose the following inputs are given to the program: 3,5
# Then, the output of the program should be:
# [[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 2, 4, 6, 8]]
# Hints: use split() method
class TwoDimensionalArray(object):
    def __init__(self):
        self.result = []

    def input2DimensionalNumber(self):
        dimenionalNumber = [int(x) for x in input("Enter list number: ").split(",")]
        self.row = dimenionalNumber[0]
        self.column = dimenionalNumber[1]

    def generate2DimensionalArray(self):
        for i in range(0, int(self.row)):
            self.result.append([i * j for j in range(0, int(self.column))])

    def printDimensionalArray(self):
        print(self.result)


obj = TwoDimensionalArray()
obj.input2DimensionalNumber()
obj.generate2DimensionalArray()
obj.printDimensionalArray()

# Question 8
# Write a program that accepts a comma separated sequence of words as input and prints the
# words in a comma-separated sequence after sorting them alphabetically.
# Example:
# Suppose the following input is supplied to the program:
# without,hello,bag,world
# Then, the output should be:
# bag,hello,without,world
# Hints: use sort() method
words = [x for x in input("Enter list words: ").split(",")]
words.sort()
print(",".join(words))

# Question 9
# Write a program that accepts sequence of lines as input and prints the lines after making all
# characters in the sentence capitalized.
# Example:
# Suppose the following input is supplied to the program:
# Hello world
# Practice makes perfect
# Then, the output should be:
# HELLO WORLD
# PRACTICE MAKES PERFECT
# Hints: use upper() method

lines = []
print("Enter words which you want to upper: ")
while True:
    line = input()
    if line:
        lines.append(line.upper())
    else:
        break
for line in lines:
    print(line)

# Question 10:
# Write a program that accepts a sequence of whitespace separated words as input and prints
# the words after removing all duplicate words and sorting them alphanumerically.
# Example:
# Suppose the following input is supplied to the program:
#  hello world and practice makes perfect and hello world again
# Then, the output should be:
#  again and hello makes perfect practice world
# Hints:
# We use set container to remove duplicated data automatically and then use sorted() to sort the
# data.
words = input("Enter words which you want to sort: ").split(" ")
print(sorted(set(words)))

# Question 11:
# Write a program which accepts a sequence of comma separated 4 digit binary numbers as its
# input and then check whether they are divisible by 5 or not. The numbers that are divisible by 5
# are to be printed in a comma separated sequence.
# Example:
# 0100,0011,1010,1001
# Then the output should be:
# 1010
# Hints: use int() function to convert binary numbers to decimal numbers

binaryNumbers = input("Enter list of binary number which you want to divide 5: ").split(",")
result = [x for x in binaryNumbers if int(x) % 5 == 0]
print(result)

# Question 12:
# Write a program, which will find all such numbers between 1000 and 3000 (both included) such
# that each digit of the number is an even number.
# The numbers obtained should be printed in a comma-separated sequence on a single line.
# Hints: use str() function to convert integers to string
result = []
numberCount = len(str(abs(3001)))
for i in range(1000, 3001):
    check = True
    for numbericalOrder in range(numberCount):
        if int(str(i)[numbericalOrder]) % 2 != 0:
            check = False
    if check:
        result.append(str(i))
print(",".join(result))

# Quesion 13:
# Write a program that accepts a sentence and calculate the number of letters and digits.
# Example:
# Suppose the following input is supplied to the program: hello world! 123
# Then, the output should be:
# LETTERS 10
# DIGITS 3
text = input("Enter letters and digits: ")
number = sum(char.isdigit() for char in text)
letter = sum(char.isalpha() for char in text)
print("Number: " + number + "\nLetter: " + letter)
# spaces  = sum(c.isspace() for c in s)

# Question 14:
# Write a program that accepts a sentence and calculate the number of upper case letters and
# lower case letters.
# Example:
# Suppose the following input is supplied to the program:
# Hello world!
# Then, the output should be:
# UPPER CASE 1
# LOWER CASE 9
text = input("Enter lower case letters and upper case letters: ")
lowerCase = sum(char.islower() for char in text)
upperCase = sum(char.isupper() for char in text)
print("Lower: " + str(lowerCase) + "\nUpper: " + str(upperCase))

# Question 5
# Write a program that computes the value of a+aa+aaa+aaaa with a given digit as the value of a.
# Example:
# Suppose the following input is supplied to the program: 9
# Then, the output should be:11106

# C1:
value = input("Enter value: ")
n1 = value * 1
n2 = value * 2
n3 = value * 3
n4 = value * 4
print(int(n1) + int(n2) + int(n3) + int(n4))

# C2:
value = input()
n1 = int("%s" % value)
n2 = int("%s%s" % (value, value))
n3 = int("%s%s%s" % (value, value, value))
n4 = int("%s%s%s%s" % (value, value, value, value))
print(n1 + n2 + n3 + n4)
