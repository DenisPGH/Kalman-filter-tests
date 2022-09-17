from filterpy.kalman import predict, update
import math
import numpy as np
from numpy.random import randn
 ### Tracking a Dog
def compute_dog_data(z_var, process_var, count=1, dt=1.):
    "returns track, measurements 1D ndarrays"
    x, vel = 0., 1.
    z_std = math.sqrt(z_var)
    p_std = math.sqrt(process_var)
    xs, zs = [], []
    for _ in range(count):
        v = vel + (randn() * p_std)
        x += v*dt
        xs.append(x)
        zs.append(x + randn() * z_std)
    return np.array(xs), np.array(zs)

### Predict Step -  Design State Variable
x = np.array([[10.0], # position 10m
              [4.5]]) # speed 4.5 m/s     or write so  x = np.array([[10., 4.5]]).T


####### jUST test multiplication
A = np.array([[1, 2], [3, 4]])
x = np.array([[10.0], [4.5]])

# matrix multiply
print(np.dot(A, x))

# alternative matrix multiply)
print(A @ x)
print()

x = np.array([[10.0, 4.5]]).T
print(A @ x)
print()

x = np.array([10.0, 4.5])
print(A @ x)
####### jUST test


### Predict step- Design State Covariance ~ P #############################
P = np.diag([500., 49.])   #velositi=7**2 =49 , estimated start position =500 sq.m
print(P)


########### Predict step - Design the Process Model ################################
from collections import namedtuple
gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: f'𝒩(μ={s[0]:.3f}, 𝜎²={s[1]:.3f})'

def predict(pos, movement):
    return gaussian(pos.mean + movement.mean,
                    pos.var + movement.var)

from scipy.linalg import solve
A = np.array([[2, 3],[4, -1]])
b = np.array([[8], [2]])
x = solve(A, b)
print(f"test lin= {x}")
# declare F= state transition matrix (function)
dt = 0.1
F = np.array([[1, dt],
              [0, 1]])

# test my prediction model
from filterpy.kalman import predict

x = np.array([10.0, 4.5])
P = np.diag([500, 49])
F = np.array([[1, dt], [0, 1]])

# Q is the process noise
x, P = predict(x=x, P=P, F=F, Q=0)
print('x =', x)
for _ in range(4):
    x, P = predict(x=x, P=P, F=F, Q=0)
    print('x =', x)

print(P)
##################### plot the prediction
""" The initial value is in solid red, and the prior (prediction) is in dashed black. """
from filterpy.stats import plot_covariance_ellipse
import matplotlib.pyplot as plt

dt = 0.3
F = np.array([[1, dt], [0, 1]])
x = np.array([10.0, 4.5])
P = np.diag([500, 500])
plot_covariance_ellipse(x, P, edgecolor='r')
x, P = predict(x, P, F, Q=0)
plot_covariance_ellipse(x, P, edgecolor='k', ls='dashed')
plt.show()

########### Prediction step - Design Process Noise ################################
from filterpy.common import Q_discrete_white_noise
Q = Q_discrete_white_noise(dim=2, dt=1., var=2.35) # white proccess noise dim=dim of matrix, dt=time step, var=variance
print(Q)

############# Prediction step - Design the Control Function #################################
B = 0.  # my dog doesn't listen to me!
u = 0
x, P = predict(x, P, F, Q, B, u)
print('x =', x)
print('P =', P)
"""
Prediction: Summary
Your job as a designer is to specify the matrices for

x, P : the state and covariance
F, Q : the process model and noise covariance
B, u: Optionally, the control input and function
"""

################ UPDATE STEP -- Design the Measurement Function #################################

H = np.array([[1., 0.]])  # [position,velocity] # need only pos=1, not velocity=0

################ UPDATE STEP -- Design the Measurement #################################
""" z=[z].T one sensor,     z=[z1,z2,z3].T  poveche sensors"""

R = np.array([[5.]]) #one sensor, if two R=[[5,0],[0,2]] diagonal

######## test update ####################################

from filterpy.kalman import update
z = 1.
x, P = update(x, P, z, R, H)
print('x =', x)




