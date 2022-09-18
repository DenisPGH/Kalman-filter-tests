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