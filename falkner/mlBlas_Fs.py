'''
Falkner and Blasius Equations Solution with Neural Network Pure Implementation
'''

import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam
from time import process_time
import pickle
import matplotlib.pyplot as plt

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]


def swish(x):
    "see https://arxiv.org/pdf/1710.05941.pdf"
    return x / (1.0 + np.exp(-x))


def f(params, inputs):
    "Neural network functions"
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = swish(outputs)    
    return outputs

t1_start = process_time()    
# Here is our initial guess of params:
params = init_random_params(0.1, layer_sizes=[1, 8, 1])

# Derivatives
fp = elementwise_grad(f, 1)
fpp = elementwise_grad(fp, 1)
fppp = elementwise_grad(fpp, 1)

eta = np.linspace(0, 10).reshape((-1, 1))
beta = -0.198
alpha = 1.0
##  the falkner-scan & Blasus equation. 
## set alpha = 0.5, beta = 0.0 for Blasius
## set alpha = 1.0, and various beta for Falkner-Skan
# This is the function we seek to minimize
def objective(params, step):
    # These should all be zero at the solution
    # f''' + 0.5 f'' f = 0
    # zeq = fppp(params, eta) + 0.5 * f(params, eta) * fpp(params, eta)
    zeq = fppp(params, eta) +  alpha *f(params, eta) * fpp(params, eta) + beta*(1. - fp(params, eta)**2)
    bc0 = f(params, 0.0)  # equal to zero at solution
    bc1 = fp(params, 0.0)  # equal to zero at solution
    bc2 = fp(params, 10.0) - 1.0 # this is the one at "infinity"
    return np.mean(zeq**2) + bc0**2 + bc1**2 + bc2**2

def callback(params, step, g):
    if step % 1000 == 0:
        print("Iteration {0:3d} objective {1}".format(step,
                                                      objective(params, step)))

params = adam(grad(objective), params,
              step_size=0.001, num_iters=10000, callback=callback) 

print('f(0) = {}'.format(f(params, 0.0)))
print('fp(0) = {}'.format(fp(params, 0.0)))
print('fp(10) = {}'.format(fp(params, 10.0)))
print('fpp(0) = {}'.format(fpp(params, 0.0)))
pickle.dump(eta, open('eta.pk', 'wb'), protocol=4)
pickle.dump(f(params, eta), open('feta.pk', 'wb'), protocol=4)
plt.plot(eta, fp(params, eta), 'b',  eta, fpp(params, eta), 'r')
plt.xlabel('$\eta$')
plt.ylabel('$fp & fpp(\eta)$')
plt.xlim([0, 10])
plt.ylim([0, 1.2])
plt.show()
plt.savefig('images/nn-blasius.png')
t1_stop = process_time()
print("elapsed time in secs: ", t1_stop-t1_start)
## results , beta = -0.198, fpp(0) = 0.070586, ; beta = 1.0, fpp0 = 1.233588:
## beta = 1.3333, fpp0 = 1.401223 : beta = 2.0, fpp0 = 1.68840: beta = 0.0, fppo = 0.4700
## beta = 0.5, fpp0 0.9282961 : beta = -0.1, fpp0 = 0.32012 : beta = 1.5, fpp0 = 1.47808
