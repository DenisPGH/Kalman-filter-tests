from filterpy.common import Q_discrete_white_noise
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.stats import plot_covariance

from One_dim_kal_2 import plot_measurements, plot_filter
import matplotlib.pyplot as plt

from a6_1_Multivar_kalman_filter import compute_dog_data
from a6_2_Multi_kalman_filter import pos_vel_filter, run


#### Implement Kalman filter without filterpy #########################################

def plot_track(ps, actual, zs, cov, std_scale=1,
               plot_P=True, y_lim=None,
               xlabel='time', ylabel='position',
               title='Kalman Filter'):

    count = len(zs)
    zs = np.asarray(zs)

    cov = np.asarray(cov)
    std = std_scale * np.sqrt(cov[:, 0, 0])
    std_top = np.minimum(actual+std, [count + 10])
    std_btm = np.maximum(actual-std, [-50])

    std_top = actual + std
    std_btm = actual - std

    #plot_track(actual)
    #plot_track(actual, actual, zs, cov)
    #print(actual) # tracking the path
    plt.plot(actual, linestyle='-', color='red', lw=2, alpha=0.6)
    plot_measurements(range(1, count + 1), zs)
    plot_filter(range(1, count + 1), ps)

    plt.plot(std_top, linestyle=':', color='k', lw=1, alpha=0.4)
    plt.plot(std_btm, linestyle=':', color='k', lw=1, alpha=0.4)
    plt.fill_between(range(len(std_top)), std_top, std_btm,
                     facecolor='yellow', alpha=0.2, interpolate=True)
    plt.legend(loc=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if y_lim is not None:
        plt.ylim(y_lim)
    else:
        plt.ylim((-50, count + 10))

    plt.xlim((0, count))
    plt.title(title)
    plt.show()

    if plot_P:
        ax = plt.subplot(121)
        ax.set_title(r"$\sigma^2_x$ (pos variance)")
        plot_covariance(cov, (0, 0))
        ax = plt.subplot(122)
        ax.set_title(r"$\sigma^2_\dot{x}$ (vel variance)")
        plot_covariance(cov, (1, 1))
        plt.show()
    plt.grid()

dt = 1.
R_var = 10
Q_var = 0.01
x = np.array([[10.0, 4.5]]).T
P = np.diag([500, 49])
F = np.array([[1, dt],
              [0,  1]])
H = np.array([[1., 0.]])
R = np.array([[R_var]])
Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_var)

from scipy.linalg import inv

count = 50
track, zs = compute_dog_data(R_var, Q_var, count)
xs, cov = [], []
for z in zs:
    # predict
    x = F @ x
    P = F @ P @ F.T + Q

    # update
    S = H @ P @ H.T + R
    K = P @ H.T @ inv(S)
    y = z - H @ x
    x += K @ y
    P = P - K @ H @ P

    xs.append(x)
    cov.append(P)

xs, cov = np.array(xs), np.array(cov)
###plot_track(xs[:, 0], track, zs, cov, plot_P=False)


##############  Exercise: Show Effect of Hidden Variables #################################

from math import sqrt
from numpy.random import randn


def univariate_filter(x0, P, R, Q):
    f = KalmanFilter(dim_x=1, dim_z=1, dim_u=1)
    f.x = np.array([[x0]])
    f.P *= P
    f.H = np.array([[1.]])
    f.F = np.array([[1.]])
    f.B = np.array([[1.]])
    f.Q *= Q
    f.R *= R
    return f


def plot_1d_2d(xs, xs1d, xs2d):
    plt.plot(xs1d, label='1D Filter')
    plt.scatter(range(len(xs2d)), xs2d, c='r', alpha=0.7, label='2D Filter')
    plt.plot(xs, ls='--', color='k', lw=1, label='track')
    plt.title('State')
    plt.legend(loc=4)
    plt.show()


def compare_1D_2D(x0, P, R, Q, vel, u=None):
    # storage for filter output
    xs, xs1, xs2 = [], [], []

    # 1d KalmanFilter
    f1D = univariate_filter(x0, P, R, Q)

    # 2D Kalman filter
    f2D = pos_vel_filter(x=(x0, vel), P=P, R=R, Q=0)
    if np.isscalar(u):
        u = [u]
    pos = 0  # true position
    for i in range(100):
        pos += vel
        xs.append(pos)

        # control input u - discussed below
        f1D.predict(u=u)
        f2D.predict()

        z = pos + randn() * sqrt(R)  # measurement
        f1D.update(z)
        f2D.update(z)

        xs1.append(f1D.x[0])
        xs2.append(f2D.x[0])
    plt.figure()
    plot_1d_2d(xs, xs1, xs2)

#################### tests different settings ##################################
#compare_1D_2D(x0=0, P=50., R=5., Q=.02, vel=1.)

#compare_1D_2D(x0=0, P=50., R=5., Q=.02, vel=1., u=1.)
#compare_1D_2D(x0=0, P=50., R=5., Q=.02, vel=-2., u=1.)

################ Adjusting the Filter ##################################

from numpy.random import seed
seed(2)
trk, zs = compute_dog_data(z_var=225, process_var=.02, count=50)

# run(track=trk, zs=zs, R=225, Q=200, P=P, plot_P=False,
#     title='R_var = 225 $m^2$, Q_var = 20 $m^2$') # big Q not trust the prediction
# run(track=trk, zs=zs, R=225, Q=.02, P=P, plot_P=False,
#     title='R_var = 225 $m^2$, Q_var = 0.02 $m^2$') # small Q we trust the prediction
#
#
# run(track=trk, zs=zs, R=10000, Q=.2, P=P, plot_P=False,
#     title='R=$10,000\, m^2$, Q=$.2\, m^2$') # very big R , measurment noise
#
var=27.5
# run(track=trk, zs=zs, R=var, Q=.02, P=500., plot_P=True,
#     title='$P=500\, m^2$') # with big P
#
# run(track=trk, zs=zs, R=var, Q=.02, P=1., plot_P=True,
#     title='$P=1\, m^2$') # small P


x = np.array([100., 0.])
run(track=trk, zs=zs, R=var, Q=.02, P=1., x0=x,
    plot_P=False, title='$P=1\, m^2$') # wrong estimate start pos(100,0), and it is (0,0), and small P

