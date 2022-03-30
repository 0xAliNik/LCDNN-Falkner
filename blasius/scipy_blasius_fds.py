import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

## define eta limits
x0 = 0.0
xn = 6.0

## define bounday conditions
alpha = 0.0
beta = 1.0

## dfine step size h
N = 100
X = np.linspace(x0, xn, N)
h = (xn - x0) / (N-1 )

## def functions for calculation

def Yppp(x, y, ypp):
    '''define y'' = 3*y*y' '''
    return -0.5 * y * ypp

def residuals(y):
    '''When we have the right values of y, this function will be zero.'''

    res = np.zeros(y.shape)

    res[0] = y[0] - alpha
    res[1] = y[1] - 0.0

    for i in range(2, N - 2):
        x = X[i]
        YPP = (y[i - 1] - 2 * y[i] + y[i + 1]) / h**2
        YP = (y[i + 1] - y[i - 1]) / (2 * h)
        YPPP = (y[i+2] -2.* y[i+1] + 2.* y[i-1] - y[i-2])/(2.*h**3)

        res[i] = YPPP - Yppp(x, y[i], YPP)
    
    res[-1] = y[-1] - y[-2] - beta*h
    res[-2] = y[-2] - y[-3] - beta*h
    return res

# we need an initial guess
init = alpha + (beta - alpha) / (xn - x0) * X

## Thes solver. Finds root for F(x) = 0
Y = fsolve(residuals, init)

zz0 = (Y[2] - 2.*Y[1] + Y[0])/h**2
print("Blausius fpp0 :", zz0)

##Plotting functions for pyplot
r = []
z = []
zz = []
z0 = 0.0
r = [x0] + r
z = [z0] + z
zz = [zz0] + zz
for i in range(2, N ):
   if i < N -1 :
     YP = (Y[i + 1] - Y[i - 1]) / (2 * h)
     YPP = (Y[i - 1] - 2 * Y[i] + Y[i + 1]) / h**2
   else :
     YP = 1.0
     YPP = 0.0
   r = [X[i]] + r        
   z = [YP]+ z
   zz = [YPP] + zz
r = [xn] + r
z = [1.0]+ z
zz = [0.0] + zz
r.reverse()
z.reverse()
zz.reverse()
plt.xlabel('$\eta$')
plt.plot(X, Y, 'b', label = "f")
plt.plot(r,z, 'g', label = "fp")
plt.plot(r,zz, 'r', label = "fpp")
plt.title("Blausius Finite Diff Sol")
plt.legend()
plt.show()
# plt.savefig('images/bvp-nonlinear-1.png')
