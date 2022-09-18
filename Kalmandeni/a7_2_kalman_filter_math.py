####################### Using FilterPy to Compute Q ###############################

from filterpy.common import Q_continuous_white_noise
from filterpy.common import Q_discrete_white_noise

Q = Q_continuous_white_noise(dim=2, dt=1, spectral_density=1)
print(Q)

Q = Q_continuous_white_noise(dim=3, dt=1, spectral_density=1)
print(Q)

print('NEWWWWWWWWWWWWWW with 2')
Q = Q_discrete_white_noise(2, var=1.)
print(Q)

print('NEWWWWWWWWWWWWWW with 3')
Q = Q_discrete_white_noise(3, var=1.)
print(Q)


##################### Simplification of Q ###############################

import numpy as np

np.set_printoptions(precision=8)
Q = Q_continuous_white_noise(
    dim=3, dt=0.05, spectral_density=1)
print(np.around(Q,3))
np.set_printoptions(precision=3)


######################### Euler's Method ###################################
"""  We happen to know the exact answer is 
 because we solved it earlier, but for an arbitrary ODE we will not know the exact solution. 
 In general all we know is the derivative of the equation, which is equal to the slope.
  We also know the initial value: at t=0 ,y=1 . If we know these two pieces of information 
  we can predict the value at y(t=1) using the slope at t=0  and the value of y(0). 
  I've plotted this below. """
import matplotlib.pyplot as plt

# t = np.linspace(-1, 1, 10)
# plt.plot(t, np.exp(t))
# t = np.linspace(-1, 1, 2)
# plt.plot(t,t+1, ls='--', c='k')
# plt.grid()
# plt.show()

def euler(t, tmax, y, dx, step=1.):
    ys = []
    while t < tmax:
        y = y + step*dx(t, y)
        ys.append(y)
        t +=step
    return ys


def dx(t, y): return y

print(euler(0, 1, 1, dx, step=1.)[-1])
print(euler(0, 2, 1, dx, step=1.)[-1])


ys = euler(0, 4, 1, dx, step=0.00001)
# plt.subplot(1,2,1)
# plt.title('Computed')
# plt.plot(np.linspace(0, 4, len(ys)),ys)
# plt.subplot(1,2,2)
# t = np.linspace(0, 4, 20)
# plt.title('Exact')
#plt.plot(t, np.exp(t))
#plt.show()

print('exact answer=', np.exp(4))
print('euler answer=', ys[-1])
print('difference =', np.exp(4) - ys[-1])
print('iterations =', len(ys))


########################  Runge Kutta Methods ##########################################


def runge_kutta4(y, x, dx, f):
    """computes 4th order Runge-Kutta for dy/dx.
    y is the initial value for y
    x is the initial value for x
    dx is the difference in x (e.g. the time step)
    f is a callable function (y, x) that you supply
    to compute dy/dx for the specified values.
    """

    k1 = dx * f(y, x)
    k2 = dx * f(y + 0.5 * k1, x + 0.5 * dx)
    k3 = dx * f(y + 0.5 * k2, x + 0.5 * dx)
    k4 = dx * f(y + k3, x + dx)

    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.



import math
import numpy as np
t = 0.
y = 1.
dt = .1

ys, ts = [], []

def func(y,t):
    return t*math.sqrt(y)

while t <= 10:
    y = runge_kutta4(y, t, dt, func)
    t += dt
    ys.append(y)
    ts.append(t)

exact = [(t**2 + 4)**2 / 16. for t in ts]
plt.plot(ts, ys)
plt.plot(ts, exact)
plt.grid()
plt.show()

error = np.array(exact) - np.array(ys)
print(f"max error {max(error):.5f}")