import numpy as np
class MonteCarloIntergration:
    def __init__(self,func_f,func_p):
        self.func_f =  func_f
        self.func_p = func_p
    def solve(self,num_samples):
        '''

        :param num_samples:
        :return:fm
        '''
        samples = self.func_p(num_samples)
        vfunc_f = lambda x:self.func_f(x)
        vfunc_f = np.vectorize(vfunc_f)
        y = vfunc_f(samples)
        return np.sum(y)/num_samples
def func_f(x):
    return x**2*np.sqrt(2*np.pi)
def func_p(n):
    return np.random.standard_normal(int(n))
num_samples = 1e6

# 使用蒙特卡罗积分法进行求解
monte_carlo_integration = MonteCarloIntergration(func_f, func_p)
result = monte_carlo_integration.solve(num_samples)
print("抽样样本数量:", num_samples)
print("近似解:", result)