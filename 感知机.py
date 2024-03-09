import matplotlib.pyplot as plt
import numpy as np

x = np.array([[-1,-1],[0,0],[2,3],[3,2]])
y = np.array([1,1,-1,-1])
w = np.array([0,0])
b = 0.0
eta = 1
def sign(x,b,w):
    return np.sign(np.dot(x,w) + b)
def feeling_manchemis_fit(x,y,w,b):
    check = False
    while not check:
        check = True
        for xi,yi in zip(x,y):
            if sign(xi,b,w)*yi <= 0:
                w += eta*xi*yi
                b += eta*yi
                check = False
    return w,b
w,b = feeling_manchemis_fit(x,y,w,b)
xx = np.linspace(-2,4,100)
yy = -1/w[1]*(w[0]*xx + b)

plt.figure()
plt.scatter(x[:,0],x[:,1],c=y)
plt.plot(xx,yy)
plt.show()

