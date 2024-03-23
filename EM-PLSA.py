import numpy as np
import pandas as pd
X = [[0,0,1,1,0,0,0,0,0],
     [0,0,0,0,0,1,0,0,1],
     [0,1,0,0,0,0,0,1,0],
     [0,0,0,0,0,0,1,0,1],
     [1,0,0,0,0,1,0,0,0],
     [1,1,1,1,1,1,1,1,1],
     [1,0,1,0,0,0,0,0,0],
     [0,0,0,0,0,0,1,0,1],
     [0,0,0,0,0,2,0,0,1],
     [1,0,1,0,0,0,0,1,0],
     [0,0,0,1,1,0,0,0,0]]
X = np.array(X)
class PLSA:
    def __init__(self, K, max_iter):
        self.K = K
        self.max_iter = max_iter

    def fit(self, X):
        n_d, n_w = X.shape

        # P(z|w,d)
        p_z_dw = np.zeros((n_d, n_w, self.K))

        # P(z|d)
        p_z_d = np.random.rand(n_d, self.K)

        # P(w|z)
        p_w_z = np.random.rand(self.K, n_w)

        for i_iter in range(self.max_iter):
            # E step
            for di in range(n_d):
                for wi in range(n_w):
                    sum_zk = np.zeros((self.K))
                    for zi in range(self.K):
                        sum_zk[zi] = p_z_d[di, zi] * p_w_z[zi, wi]
                    sum1 = np.sum(sum_zk)
                    if sum1 == 0:
                        sum1 = 1
                    for zi in range(self.K):
                        p_z_dw[di, wi, zi] = sum_zk[zi] / sum1

            # M step

            # update P(z|d)
            for di in range(n_d):
                for zi in range(self.K):
                    sum1 = 0.
                    sum2 = 0.

                    for wi in range(n_w):
                        sum1 = sum1 + X[di, wi] * p_z_dw[di, wi, zi]
                        sum2 = sum2 + X[di, wi]

                    if sum2 == 0:
                        sum2 = 1
                    p_z_d[di, zi] = sum1 / sum2

            # update P(w|z)
            for zi in range(self.K):
                sum2 = np.zeros((n_w))
                for wi in range(n_w):
                    for di in range(n_d):
                        sum2[wi] = sum2[wi] + X[di, wi] * p_z_dw[di, wi, zi]
                sum1 = np.sum(sum2)
                if sum1 == 0:
                    sum1 = 1
                    for wi in range(n_w):
                        p_w_z[zi, wi] = sum2[wi] / sum1

        return pd.DataFrame(p_w_z), pd.DataFrame(p_z_d)

model = PLSA(2, 100)
p_w_z, p_z_d = model.fit(X)
print(p_w_z.applymap(lambda x: '%.2f'%x))
print(p_z_d.applymap(lambda x: '%.2f'%x))