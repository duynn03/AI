import numpy as np

# Question 1: Add the following two NumPy arrays and Modify a result array by calculating the square root of each element
print("Question 1: ")
A = np.array([[4, 8],
              [1, 2]])
B = np.array([[1, 3],
              [2, 6]])
C = A + B
print(C)
C = np.square(C)
print(C)

# Question 2: Split the array into four equal-sized sub-arrays
print("\nQuestion 2: ")
A = np.array(range(12)) + 1
print(A)
result = np.split(A, 4)
print(result)

# Question 3: Ex 3.Sort following NumPy array

# https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
print("\nQuestion 3: ")
A = np.array([[8, 1, 9, 6],
              [4, 2, 0, 3],
              [1, 5, 6, 9]])
# 3.1- by the second row
print("Row: ")
secondRow = A[1, :]
print(secondRow)
indiceRow = secondRow.argsort()
print(indiceRow)
print(A[:, indiceRow])

# 3.2-by the second column
print("\nColumn: ")
secondColumn = A[:, 3]
print(secondColumn)
indiceColumn = secondColumn.argsort()
print(indiceColumn)
print(A[indiceColumn, :])
