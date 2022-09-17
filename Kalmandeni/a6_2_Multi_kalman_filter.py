################ Implementing the Kalman Filter ##############################

from filterpy.kalman import KalmanFilter
from filterpy.stats import plot_covariance

from One_dim_kal_2 import plot_measurements, plot_filter

dog_filter = KalmanFilter(dim_x=2, dim_z=1) # two var(pos,velocity) => dim_x=2,  dim_z=1(one measurments)
print('x = ', dog_filter.x.T)
print('R = ', dog_filter.R)
print('Q = \n', dog_filter.Q)

from filterpy.common import Q_discrete_white_noise
import numpy as np
import matplotlib.pyplot as plt
import math

from numpy.random import randn



def pos_vel_filter(x, P, R, Q=0., dt=1.0):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([x[0], x[1]])  # location and velocity
    kf.F = np.array([[1., dt],
                     [0., 1.]])  # state transition matrix
    kf.H = np.array([[1., 0]])  # Measurement function
    kf.R *= R  # measurement uncertainty
    if np.isscalar(P):
        kf.P *= P  # covariance matrix
    else:
        kf.P[:] = P  # [:] makes deep copy
    if np.isscalar(Q):
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
    else:
        kf.Q[:] = Q
    return kf
##### implement kalman filter here with the start values #######################
dt = .1
x = np.array([0., 0.])
kf = pos_vel_filter(x, P=500, R=5, Q=0.1, dt=dt)
print(f"resultat {kf}")

### run code for use the Kalman filter ########################
from filterpy.common import Saver
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
    print(actual) # tracking the path
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


def run(x0=(0.,0.), P=500, R=0, Q=0, dt=1.0,
        track=None, zs=None,
        count=0, do_plot=True, **kwargs):
    """
    track is the actual position of the dog, zs are the
    corresponding measurements.
    """

    # Simulate dog if no data provided.
    if zs is None:
        track, zs = compute_dog_data(R, Q, count)

    # create the Kalman filter
    kf = pos_vel_filter(x0, R=R, P=P, Q=Q, dt=dt)
    s = Saver(kf)
    # run the kalman filter and store the results
    xs, cov = [], [] # or can use Saver class
    for z in zs:
        kf.predict()
        kf.update(z)
        xs.append(kf.x)
        cov.append(kf.P)
        s.save()
    print(s.x)

    xs, cov = np.array(xs), np.array(cov)
    if do_plot:
        plot_track(xs[:, 0], track, zs, cov, **kwargs)
    return xs, cov





P = np.diag([500., 49.])
#Ms, Ps = run(count=50, R=10, Q=0.01, P=P)


############# SAVER CLASS #################################
""" the class can save the result for us  xs, cov = [], [] """

# from filterpy.common import Saver
# kf = pos_vel_filter([0, .1], R=R, P=P, Q=Q, dt=1.)
# s = Saver(kf)
# for i in range(1, 6):
#     kf.predict()
#     kf.update([i])
#     s.save()  # save
#
# print(s.x)
