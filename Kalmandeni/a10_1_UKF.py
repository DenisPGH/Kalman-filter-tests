from filterpy.kalman import JulierSigmaPoints
import  numpy as np
import matplotlib.pyplot as plt

sigmas = JulierSigmaPoints(n=2, kappa=1)

def plot_sigmas(sigmas, x, cov):
    if not np.isscalar(cov):
        cov = np.atleast_2d(cov)
    pts = sigmas.sigma_points(x=x, P=cov)
    plt.scatter(pts[:, 0], pts[:, 1], s=sigmas.Wm*1000)
    plt.axis('equal')
    plt.grid()
    plt.show()


plot_sigmas(sigmas, x=[3, 17], cov=[[1, .5], [.5, 3]])

############# x=[x,x_.].T
def fx(x, dt):
    xout = np.empty_like(x)
    xout[0] = x[1] * dt + x[0]
    xout[1] = x[1]
    return xout

def hx(x):
    return x[:1] # return position [x]


from numpy.random import randn
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise

ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=1., hx=hx, fx=fx, points=sigmas)
ukf.P *= 10
ukf.R *= .5
ukf.Q = Q_discrete_white_noise(2, dt=1., var=0.03)

zs, xs = [], []
for i in range(50):
    z = i + randn() * .5
    ukf.predict()
    ukf.update(z)
    xs.append(ukf.x[0])
    zs.append(z)

plt.plot(xs)
plt.plot(zs, marker='x', ls='')
plt.grid()
plt.show()

##############  The Unscented Transform ################
"""" It then computes the mean and covariance of the transformed points. 
That mean and covariance becomes the new estimate."""