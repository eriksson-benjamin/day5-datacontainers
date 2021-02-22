import scipy.linalg as la
import scipy as sp
import numpy as np

# a) Define a matrix A
print('Problem a)')
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(f'A = \n{A}')

# b) Define a vector b
print('\nProblem b)')
b = np.array([1, 2, 3])
print(f'b = {b}')

# c) Solve the linear system of equations A x = b
print('\nProblem c)')
x_solve = la.solve(A, b)
print(f'x_solve = {x_solve}')

# d) Check that your solution is correct by plugging it into the equation
print('\nProblem d)')
test_b = np.matmul(A, x_solve)
print(f'test_b - b = {test_b - b}')

# e) Repeat steps a-d using a random 3x3 matrix B (instead of the vector b)
print('\nProblem e)')
random_a = np.random.rand(3, 3)
random_b = np.random.rand(3, 3)
x_solve = la.solve(random_a, random_b)
test_b = np.matmul(random_a, x_solve)
print(f'test_b - random_b = \n{test_b - random_b}')

# f) Solve the eigenvalue problem for the matrix A and print the eigenvalues and eigenvectors
print('\nProblem f)')
eig_vals, eig_vecs = np.linalg.eig(A)
print(f'eig_vals = {eig_vals}')
print(f'eig_vecs = \n{eig_vecs}')

# g) Calculate the inverse, determinant of A
print('\nProblem g)')
inv_A = la.inv(random_a)
det_A = la.det(random_b)
print(f'inv_A = \n{inv_A}')
print(f'det_A = {det_A}')

# h) Calculate the norm of A with different orders
print('\nProblem h)')
norm_2 = la.norm(A, ord = 2)
norm_inf = la.norm(A, ord = np.inf)
print(f'norm_2 = {norm_2}')
print(f'norm_inf = {norm_inf}')








































