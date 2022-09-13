import math
import numpy as np
import copy
import math
from numpy.random import randn
from collections import namedtuple
import matplotlib.pyplot as plt

### non linear system
def gaussian_multiply(g1, g2):
    #print(f"TEst {g1.var}, {g2.mean}, {g2.var} *{g1.mean}")
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)

def predict(pos, movement):
    return gaussian(pos.mean + movement.mean, pos.var + movement.var)

def update(prior, likelihood):
    posterior = gaussian_multiply(likelihood, prior)
    return posterior

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: f'𝒩(μ={s[0]:.3f}, 𝜎²={s[1]:.3f})'


sensor_var = 30.
process_var = 2.
pos = gaussian(100., 500.)
process_model = gaussian(1., process_var)

zs, ps = [], []

for i in range(100):
    pos = predict(pos, process_model)

    z = math.sin(i / 3.) * 2 + randn() * 1.2
    zs.append(z)

    pos = update(pos, gaussian(z, sensor_var))
    ps.append(pos.mean)

plt.plot(zs, c='r', linestyle='dashed', label='measurement')
plt.plot(ps, c='#004080', label='filter')
plt.legend(loc='best')
plt.grid()
plt.show()

"""
predict() takes several arguments, but we will only need to use these four:

predict(x, P, u, Q)
x is the state of the system. P is the variance of the system. u is the movement due to the process,
 and Q is the noise in the process. You will need to used named arguments 
 when you call predict() because most of the arguments are optional. 
 The third argument to predict() is not u.
"""

import filterpy.kalman as kf
# Let's try it for the state N(10,3)  and the movement N(1,4) . We'd expect a final position of 11 (10+1) with a variance of 7 (3+4).
a= kf.predict(x=10., P=3., u=1., Q=4.)
print(a)

""" update also takes several arguments, but for now you will be interested in these four:
update(x, P, z, R)
As before, x and P are the state and variance of the system. z is the measurement,
 and R is the measurement variance. Let's perform the last predict statement to"""

x, P = kf.predict(x=10., P=3., u=1., Q=2.**2)
print(f'{x:.3f}')

x, P = kf.update(x=x, P=P, z=12., R=3.5**2)
print(f'{x:.3f} {P:.3f}')