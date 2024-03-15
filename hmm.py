import numpy as np
class HMM:
    def forward(self,Q,V,A,B,O,PI):
        '''
        前向算法
        :param Q:状态序列
        :param V: 观测集合
        :param A:状态转移矩阵
        :param B: 生成概率矩阵
        :param O: 观测序列
        :param PI: 状态概率向量
        :return:
        '''
        N = len(Q)
        M = len(O)
        #前向矩阵,行：状态，列：时间
        alphas = np.zeros(((N,M)))

        T = M
        for t in range(T):
            index_O = V.index(O[t])
            for i in range(N):
                if t == 0:
                    alphas[i][t] = PI[0][i] * B[i][index_O]
                else:#j->i,j为遍历指标
                    alphas[i][t] = np.dot([alpha[t-1] for alpha in alphas],[a[i] for a in A])*B[i][index_O]
        P = np.sum([alpha[M-1] for alpha in alphas])
        print('alphas:')
        print(alphas)
        print('P=%5f'% P)
        return alphas
    def backward(self,Q,V,A,B,O,PI):
        '''
                后向算法
                :param Q:状态序列
                :param V: 观测集合
                :param A:状态转移矩阵
                :param B: 生成概率矩阵
                :param O: 观测序列
                :param PI: 状态概率向量
                :return:
                '''
        N = len(Q)
        M = len(O)
        # 后向矩阵,行：状态，列：时间
        betas = np.zeros(((N, M)))

        T = M
        for i in range(N):
            betas[i][M-1] = 1
        for t in range(T-2,-1,-1):
            index_O = V.index(O[t])
            for i in range(N):#遍历状态j
                betas[i][t] = np.dot(np.multiply(A[i],[b[index_O] for b in B]),[beta[t+1] for beta in betas])
        index_O = V.index(O[0])
        P = np.dot(np.multiply(PI,[b[index_O] for b in B]),[beta[0] for beta in betas])
        print('P(O||lambada)=',end='')
        for i in range(N):
            print('%.1f*%.1f*%.5f'%(PI[0][i],B[i][index_O],betas[i][0]),end='')
        print('O=%f'%P)
        print(betas)
        return betas

if __name__=='__main__':
    Q = [1,2,3]
    V = ['red','white']
    A = [[0.5,0.1,0.4],[0.3,0.5,0.2],[0.2,0.2,0.6]]
    B = [[0.5,0.5],[0.4,0.6],[0.7,0.3]]
    O = ['red','white','red','red','white','red','white','white']
    PI = [[0.2,0.3,0.4]]
    Hmm = HMM()
    alphas = Hmm.forward(Q,V,A,B,O,PI)
    betas = Hmm.backward(Q,V,A,B,O,PI)
    alpha = [x[3] for x in alphas]
    beta = [x[3] for x in betas]
    print(alpha)
    print(beta)
    result = alpha[2]*beta[2]/np.dot(alpha,beta)
    print(result)