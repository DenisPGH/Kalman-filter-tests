############ Batch processing #################################
"""
First collect your measurements into an array or list. Maybe it is in a CSV file:

zs = read_altitude_from_csv('altitude_data.csv')
Or maybe you will generate it using a generator:

zs = [some_func(i) for i in range(1000)]
Then call the batch_filter() method.

Xs, Ps, Xs_prior, Ps_prior = kfilter.batch_filter(zs)
The function takes the list of measurements, filters it,
 and returns an NumPy array of state estimates (Xs), covariance matrices (Ps),
 and the priors for the same (Xs_prior, Ps_prior).
"""


from One_dim_kal_2 import plot_measurements, plot_filter
from a6_2_Multi_kalman_filter import compute_dog_data, pos_vel_filter
import matplotlib.pyplot as plt
import numpy as np

# count = 50
# track, zs = compute_dog_data(10, .2, count)
# P = np.diag([500., 49.])
# f = pos_vel_filter(x=(0., 0.), R=3., Q=.02, P=P)
# xs, _, _, _ = f.batch_filter(zs)
#
# plot_measurements(range(1, count + 1), zs)
# plot_filter(range(1, count + 1), xs[:, 0])
# plt.legend(loc='best')
# plt.grid()
# plt.show()


############################# Smoothing the Results ##############################################
from numpy.random import seed
count = 50
seed(8923)

P = np.diag([500., 49.])
f = pos_vel_filter(x=(0., 0.), R=3., Q=.02, P=P)
track, zs = compute_dog_data(3., .02, count)
Xs, Covs, _, _ = f.batch_filter(zs) # first filtering with bacht
Ms, Ps, _, _ = f.rts_smoother(Xs, Covs) # then with rts_smoother for better performming

plot_measurements(zs)
plt.plot(Xs[:, 0], ls='--', label='Kalman Position')
plt.plot(Ms[:, 0], label='RTS Position')
plt.legend(loc=4)

#### velociti chart
plt.plot(Xs[:, 1], ls='--', label='Kalman Velocity')
plt.plot(Ms[:, 1], label='RTS Velocity')
plt.legend(loc=4)
plt.gca().axhline(1, lw=1, c='k')
plt.grid()
plt.show()