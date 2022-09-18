import numpy as np

a=np.array([[0,1],[0,0]])
b=np.array([1,3])
c=a.dot(b.T)
print(c)
""" In practice this may not be of concern to you as for the Kalman filter we normally just take the first two terms of the Taylor series. """

################## Exponential matrix #############################################
import numpy as np
from scipy.linalg import expm

dt = 0.1
A = np.array([[0, 1],
              [0, 0]])
F=expm(A*dt)
print(F)

################### Numerical solution Van Loan ########################################

from filterpy.common import van_loan_discretization

A = np.array([[0., 1.], [-1., 0.]])
G = np.array([[0.], [2.]]) # white noise scaling
F, Q = van_loan_discretization(A, G, dt=0.1)
print(F)
print(Q)

############### Continuous White Noise Model #########################################
import sympy
from sympy import (init_printing, Matrix, MatMul,
                   integrate, symbols)

init_printing(use_latex='mathjax')
dt, phi = symbols('\Delta{t} \Phi_s')
F_k = Matrix([[1, dt, dt**2/2],
              [0,  1,      dt],
              [0,  0,       1]])
Q_c = Matrix([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 1]])*phi

Q = integrate(F_k * Q_c * F_k.T, (dt, 0, dt))

# factor phi out of the matrix to make it more readable
Q = Q / phi
res=MatMul(Q, phi)
print(res)