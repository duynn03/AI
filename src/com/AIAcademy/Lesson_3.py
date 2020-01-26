import sys
from datetime import datetime

# Question 1:
# Write a Python script to display the various Date Time formats
# a) Current date and time
print("Current date and time: ", datetime.now())
# b) Current year
print("Current year: ", datetime.now().strftime("%Y"))
# c) Month of year
print("Month of year: ", datetime.now().strftime("%B"))
# d) Week number of the year
print("Week number of the year: ", datetime.now().strftime("%W"))
# e) Weekday of the week
print("Weekday of the week: ", datetime.now().strftime("%w"))
# f) Day of year
print("Day of year: ", datetime.now().strftime("%j"))
# g) Day of the month
print("Day of the month: ", datetime.now().strftime("%d"))
# h) Day of week
print("Day of week: ", datetime.now().strftime("%A"))

# Question 2:
# Write a Python program to convert a string to datetime.
# Sample String : Jan 7 2014 2:43PM
# Expected Output : 2014-07-01 14:43:00

date_input = 'Jan 7 2014 2:43PM'
result = datetime.strptime(date_input, '%b %d %Y %I:%M%p')
print("Time convert: ", result.strftime("%Y-%d-%m %H:%M:%S"))


# Question 3:
# Write a program to find the oldest date in a given list of dates. Print the
# number of seconds passed since that date to today
# For example.
# datetime_list = ['25.08.1995 00:00:00', '22.07.1999 00:00:00', '01.01.2001
# 13:42:59', '13.12.2011 01:02:03']
# Oldest date is '25.08.1995 00:00:00'
# You also need to find the format yourself.
# Hint: You can get each datetime as total seconds since 1 AD and compare
# these values to find oldest date

# Question 4:
# Write a Python function to sum all the numbers in a list.
# Sample List : (8, 2, 3, 0, 7)
# Expected Output : 20
def sum_list_number(list_input):
    """sum list number"""
    return sum(list_input)


print("Total: ", sum_list_number([8, 2, 3, 0, 7]))


# Question 5:
# Write a Python function to multiply all the numbers in a list.
# Sample List : (8, 2, 3, -1, 7)
# Expected Output : -336
def multiply_list_number(list_input):
    total = 1
    for i in list_input:
        total *= i
    return total


print(multiply_list_number([8, 2, 3, -1, 7]))

# Question 6:
# Write a Python program to reverse a string.
# Sample String : "1234abcd"
# Expected Output : "dcba4321"
# C1:
date_input = "1234abcd"
result = date_input[::-1]
print("Reverve String: ", result)

# C2:
date_input = "1234abcd"
result = ""
for i in range(1, len(date_input) + 1):
    result += date_input[-1 * i]
print("Reverse String", result)
# Question 7:
# Write a Python function that takes a list and returns a new list with unique
# elements of the first list.
# Sample List : [1,2,3,3,3,3,4,5]
# Unique List : [1, 2, 3, 4, 5]
date_input = [1, 2, 3, 3, 3, 3, 4, 5]
print("Unique list: ", list(set(date_input)))


# Question 8:
# Write a Python program to read an entire text file.
def read_file(file_path):
    try:
        file = open(file_path, 'r', encoding='utf-8')
        return file.read()
    finally:
        file.close()


path = sys.path[0] + "\\IO\\" + 'test.txt'
print("Entire text file:\n", read_file(path))


# Question 9:
# Write a Python program to read first n lines of a file
def read_first_line_in_file(file_path, n):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            headLines = [next(file) for x in range(n)]
        return headLines
    finally:
        file.close()


lineNumber = 2
print(lineNumber, " lines in file: \n", read_first_line_in_file(path, lineNumber))


# Question 10:
# Write a Python program to append text to a file and display the text.
def append_text_to_file(file_path, text):
    try:
        file = open(file_path, 'a', encoding='utf-8')
        file.write(text)
    finally:
        file.close()
    print("Append text to file:\n", read_file(file_path))


path = sys.path[0] + "\\IO\\" + 'test.txt'


# append_text_to_file(path, "\nCông việc: IT")

# Question 11:
# Write a Python program to read a file line by line and store it into a list
def read_line_by_line_in_file(file_path):
    try:
        file = open(file_path, 'r', encoding='utf-8')
        return file.readlines()
    finally:
        file.close()


print("Read line-by-line text in file:\n")
path = sys.path[0] + "\\IO\\" + 'test.txt'
for line in read_line_by_line_in_file(path):
    print(line)

# Question 12:
# Write a program to handle the exception when to access the array element
# whose index is out of bound
try:
    number_list = [i for i in range(5)]
    print(number_list[5])
except IndexError:
    print("An error occurred")
    pass

# Question 13:
# Write a program to handle the exception when try to divide by zero
try:
    print(1 / 0)
except ZeroDivisionError:
    print("An error occurred")
    pass


# Question 14:
# Adding handle exception code to this program to made your functions
# more robust to erroneous input/data.
# def example1():
#  for i in range( 3 ):
#  x = int( input( "enter a number: " ) )
#  y = int( input( "enter another number: " ) )
#  print( x, '/', y, '=', x/y )
# example1()

def example1():
    for i in range(3):
        x = int(input("enter a number: "))
        y = int(input("enter another number: "))
        try:
            print(x, '/', y, '=', x / y)
        except ZeroDivisionError:
            print("Can't divide by 0!")
        except ValueError:
            print("That doesn't look like a number!")
        except:
            print("something unexpected happend!")


example1()


# Question 15:
# Adding handle exception code to this program to made your functions
# more robust to erroneous input/data.
# def example2( L ):
#  print( "\n\nExample 2" )
#  sum = 0
#  sumOfPairs = []
#  for i in range( len( L ) ):
#  sumOfPairs.append( L[i]+L[i+1] )
#  print( "sumOfPairs = ", sumOfPairs )
# L = [ 10, 3, 5, 6, 9, 3 ]
# example2( L )
# example2( [ 10, 3, 5, 6, "NA", 3 ] )
# example3( [ 10, 3, 5, 6 ] )
def example2(L):
    print("\n\nExample 2")
    sumOfPairs = []
    for i in range(len(L)):
        try:
            sumOfPairs.append(L[i] + L[i + 1])
        except TypeError:
            print("Type Error")
        except IndexError:
            print("Index Error")
        except ValueError:
            print("Value Error")
            print("sumOfPairs = ", sumOfPairs)


example2([10, 3, 5, 6, 9, 3])
example2([10, 3, 5, 6, "NA", 3])
example2([10, 3, 5, 6])
