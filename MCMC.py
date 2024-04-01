import matplotlib.pyplot as plt
import numpy as np
class GibbsSampling:
    def __init__(self,target_dist,dim,m=1e4,n=1e5):
        '''
        Gibbs Sampling算法
        :param target_dist:目标分布
        :param dim: 变量维度
        :param m: 收敛步数
        :param n: 迭代步数
        '''
        self.target_dist = target_dist
        self.dim = dim
        self.m = int(m)
        self.n = int(n)

    def solve(self):
        all_samples = np.zeros((self.n,self.dim))
        x_0 = np.random.random(self.dim)
        for i in range(self.n):
            x = x_0 if i == 0 else all_samples[i-1]
            for k in range(self.dim):
                x[k] = self.target_dist.sample(x,k)
            all_samples[i] = x
        samples = all_samples[self.m:]
        print(samples.shape)
        dist_mean = samples.mean(0)
        dist_var = samples.var(0)
        return samples[self.m:], dist_mean, dist_var

    @staticmethod
    def visualize(samples, bins=50):
        """
        可视化展示
        :param samples: 抽取的随机样本集合
        :param bins: 频率直方图的分组个数
        """
        fig, ax = plt.subplots()
        ax.set_title('Gibbs Sampling')
        ax.hist(samples[:, 0], bins, alpha=0.7, label='$\\theta$')
        ax.hist(samples[:, 1], bins, alpha=0.7, label='$\\eta$')
        ax.set_xlim(0, 1)
        ax.legend()
        plt.show()

class TargetDistribution:
    def __init__(self):
        self.c = self.__select_prob_scaler()
    def sample(self,x,k):
        theta,eta = x
        if k == 0:
            while True:
                new_theta = np.random.uniform(0,1-eta)
                alpha = np.random.uniform()
                if (alpha*self.c) < self.__prob([new_theta,eta]):
                    return new_theta
        elif k==1:
            while True:
                new_eta = np.random.uniform(0, 1 - theta)
                alpha = np.random.uniform()
                if (alpha * self.c) < self.__prob([theta, new_eta]):
                    return new_eta

    @staticmethod
    def visualize(samples, bins=50):
        """
        可视化展示
        :param samples: 抽取的随机样本集合
        :param bins: 频率直方图的分组个数
        """
        fig, ax = plt.subplots()
        ax.set_title('Gibbs Sampling')
        ax.hist(samples[:, 0], bins, alpha=0.7, label='$\\theta$')
        ax.hist(samples[:, 1], bins, alpha=0.7, label='$\\eta$')
        ax.set_xlim(0, 1)
        ax.legend()
        plt.show()
    def __select_prob_scaler(self):
        '''
        放缩尺度
        :return:
        '''
        prob_list=[]
        step = 1e-3
        for theta in np.arange(step, 1, step):
            for eta in np.arange(step, 1 - theta + step, step):
                prob = self.__prob((theta, eta))
                prob_list.append(prob)
        searched_max_prob = max(prob_list)
        upper_bound_prob = searched_max_prob * 10
        return upper_bound_prob
    @staticmethod
    def __prob(x):
        """
        P(X = x) 的概率
        """
        theta = x[0]
        eta = x[1]
        p1 = (theta / 4 + 1 / 8) ** 14
        p2 = theta / 4
        p3 = eta / 4
        p4 = (eta / 4 + 3 / 8)
        p5 = 1 / 2 * (1 - theta - eta) ** 5
        p = (p1 * p2 * p3 * p4 * p5)
        return p
# 收敛步数
m = 1e3
# 迭代步数
n = 1e4

# 目标分布
target_dist = TargetDistribution()

# 使用 Gibbs Sampling 算法进行求解
gibbs_sampling = GibbsSampling(target_dist, 2, m, n)

samples, dist_mean, dist_var = gibbs_sampling.solve()

print(f'theta均值：{dist_mean[0]}, theta方差：{dist_var[0]}')
print(f'eta均值：{dist_mean[1]}, eta方差：{dist_var[1]}')

# 对结果进行可视化
GibbsSampling.visualize(samples, bins=20)



