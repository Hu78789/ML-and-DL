from cvxopt import matrix, solvers
import numpy as np

n = int(input("Enter the number of data points: "))

def information_point(n):
    points = []
    labels = []
    for i in range(n):
        point = [float(x) for x in input(f"Enter coordinates for point {i+1}: ").split()]
        label = float(input(f"Enter label for point {i+1}: "))
        points.append(point)
        labels.append(label)
    return points, labels

points, labels = information_point(n)

def build_matrix(labels, points, n):
    matrix1 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            tap = np.dot(points[i], points[j]) * labels[i] * labels[j]
            matrix1[i][j] = tap

    q = matrix(-1.0, (n, 1))
    G = matrix(np.eye(n) * -1)  # G should be a diagonal matrix with -1
    h = matrix(0.0, (n, 1))
    A = matrix(np.array([labels]))  # Transpose the labels list to create a row vector

    b = matrix(0.0)

    return matrix1, q, G, h, A, b

matrix1, q, G, h, A, b = build_matrix(labels=labels, points=points, n=n)
sol = solvers.qp(matrix(matrix1), q, G, h, A, b)

print("Optimal solution:")
print(np.array(sol['x']))
print("Primal objective value:", sol['primal objective'])




