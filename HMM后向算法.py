import numpy as np
Pi = np.array([0.2,0.4,0.4])
A = np.array([[0.5,0.2,0.3],
              [0.3,0.5,0.2],
              [0.2,0.3,0.5]])
O = np.array([0,1,0,1])
B = np.array([[0.5,0.5],
              [0.4,0.6],
              [0.7,0.3]])
beta = np.array([1,1,1])

for t in range(2,-1,-1):
    cur_beta = np.zeros(3)
    for i in range(3):
        cur_beta[i] = 0
        for j in range(3):
            cur_beta[i] += A[i][j]*B[j][O[t+1]]*beta[j]
    beta = cur_beta
    print(beta)
P = 0
for i in range(3):
    P+=Pi[i]*B[i][O[0]]*beta[i]


print(P)
