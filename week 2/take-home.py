import numpy as np
import requests

# 1. Define two custom numpy arrays, say A and B.
A = np.random.randint(10, size=(3, 3))
B = np.random.randint(10, size=(3, 3))

print(A)
print()
print(B)
print()

# Generate two new numpy arrays by stacking A and B vertically and horizontally.
ver = np.vstack((A, B))
hor = np.hstack((A, B))

print(ver)
print()
print(hor)
print()


# 2. Find common elements between A and B.
common = np.intersect1d(A, B)

print(common)
print()


# 3. Extract all numbers from A which are within a specific range
ind = np.where((A > 5) & (A < 10))
greater_than_five_less_than_ten = A[ind]
print(greater_than_five_less_than_ten)
print()


# 4. Filter the rows of iris_2d that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

ind = np.where((iris_2d[:,2] > 1.5) & (iris_2d[:,0] < 5.0))
filtered = iris_2d[ind]
print(filtered)
