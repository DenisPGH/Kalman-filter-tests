
import numpy as np
import copy
import math
from numpy.random import randn
from collections import namedtuple
import matplotlib.pyplot as plt

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: f'𝒩(μ={s[0]:.3f}, 𝜎²={s[1]:.3f})'

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


def plot_measurements(xs, ys=None, dt=None, color='k', lw=1, label='Measurements',
                          lines=False, **kwargs):
        """ Helper function to give a consistent way to display
        measurements in the book.
        """
        if ys is None and dt is not None:
            ys = xs
            xs = np.arange(0, len(ys) * dt, dt)

        plt.autoscale(tight=False)
        if lines:
            if ys is not None:
                return plt.plot(xs, ys, color=color, lw=lw, ls='--', label=label, **kwargs)
            else:
                return plt.plot(xs, color=color, lw=lw, ls='--', label=label, **kwargs)
        else:
            if ys is not None:
                return plt.scatter(xs, ys, edgecolor=color, facecolor='none',
                                   lw=2, label=label, **kwargs),
            else:
                return plt.scatter(range(len(xs)), xs, edgecolor=color, facecolor='none',
                                   lw=2, label=label, **kwargs)



def plot_filter(xs, ys=None, dt=None, c='C0', label='Filter', var=None, **kwargs):
        """ plot result of KF with color `c`, optionally displaying the variance
        of `xs`. Returns the list of lines generated by plt.plot()"""

        if ys is None and dt is not None:
            ys = xs
            xs = np.arange(0, len(ys) * dt, dt)
        if ys is None:
            ys = xs
            xs = range(len(ys))

        lines = plt.plot(xs, ys, color=c, label=label, **kwargs)
        if var is None:
            return lines

        var = np.asarray(var)
        std = np.sqrt(var)
        std_top = ys + std
        std_btm = ys - std

        plt.plot(xs, ys + std, linestyle=':', color='k', lw=2)
        plt.plot(xs, ys - std, linestyle=':', color='k', lw=2)
        plt.fill_between(xs, std_btm, std_top,
                         facecolor='yellow', alpha=0.2)

        return lines

def plot_predictions(p, rng=None, label='Prediction'):
        if rng is None:
            rng = range(len(p))
        plt.scatter(rng, p, marker='v', s=40, edgecolor='r',
                    facecolor='None', lw=2, label=label)

def show_legend():
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


class DogSimulation(object):

    def __init__(self, x0=0.0, velocity=1.0,
                 measurement_var=0.0, process_var=0.0):
        """ x0 - initial position
            velocity - (+=right, -=left)
            measurement_variance - variance in measurement m^2
            process_variance - variance in process (m/s)^2
        """
        self.x = x0
        self.velocity = velocity
        self.measurement_noise = math.sqrt(measurement_var)
        self.process_noise = math.sqrt(process_var)


    def move(self, dt=1.0):
        '''Compute new position of the dog assuming `dt` seconds have
        passed since the last update.'''
        # compute new position based on velocity. Add in some
        # process noise
        velocity = self.velocity + randn() * self.process_noise * dt
        self.x += velocity * dt


    def sense_position(self):
        # simulate measuring the position with noise
        return self.x + randn() * self.measurement_noise


    def move_and_sense(self, dt=1.0):
        self.move(dt)
        x = copy.deepcopy(self.x)
        return x, self.sense_position()


    def run_simulation(self, dt=1, count=1):
        """ simulate the dog moving over a period of time.
        **Returns**
        data : np.array[float, float]
            2D array, first column contains actual position of dog,
            second column contains the measurement of that position
        """
        return np.array([self.move_and_sense(dt) for i in range(count)])


np.random.seed(13)
process_var = 1.  # variance in the dog's movement error
sensor_var = 2.  # variance in the sensor error
x = gaussian(0., 20. ** 2)  # dog's start position, N(0, 20**2)
velocity = 1
dt = 1.  # time step in seconds
process_model = gaussian(velocity * dt, process_var)  # displacement to add to x, how we think the systems(dog) moves throught the time

# simulate dog and get measurements
dog = DogSimulation(
    x0=x.mean,
    velocity=process_model.mean,
    measurement_var=sensor_var,
    process_var=process_model.var)

# create list of measurements
zs = [dog.move_and_sense() for _ in range(10)]
print(f'PREDICT\t\t\t                        UPDATE')
print('     x      var\t\t  z\t    x      var')

# perform Kalman filter on measurement z, only 5 lines-Actual Fitler for one input varaiable
for z in zs:
    prior = predict(x, process_model)
    likelihood = gaussian(z[1], sensor_var)
    x = update(prior, likelihood)
    print(f"{prior},  {z[1]:.2f},    {x},")


print(f'final estimate:        {x.mean:10.3f}')
print(f'actual final position: {dog.x:10.3f}')


########################################################
print('NEW TEST   ###########################')
process_var = 2.
sensor_var = 4.5
x = gaussian(0., 400.)
process_model = gaussian(1., process_var)
N = 25

dog = DogSimulation(x.mean, process_model.mean, sensor_var, process_var)
zs = [dog.move_and_sense() for _ in range(N)]

xs, priors = np.zeros((N, 2)), np.zeros((N, 2))
for i, z in enumerate(zs):
    prior = predict(x, process_model)
    x = update(prior, gaussian(z[1], sensor_var))
    priors[i] = prior
    print(f"{prior},  {x}")

    xs[i] = x

### visual

fig, ax = plt.subplots()
zzs=np.array(zs)
x,y=zzs.T
filt=np.array(xs)
x_f,y_f=filt.T


#plot_measurements(zs)
#plt.scatter(x,y,color=f'blue', s=30)
#plt.scatter(x_f,y_f,color=f'red', s=30)
#plot_filter(xs[:, 0], var=priors[:, 1])
#plot_predictions(priors[:, 0])
# show_legend()
#plt.show()
#kf_internal.print_variance(xs)
#print(xs)

#################################################################

def update_primer(prior, measurement):
    x, P = prior  # mean and variance of prior
    z, R = measurement  # mean and variance of measurement

    y = z - x  # residual
    K = P / (P + R)  # Kalman gain

    x = x + K * y  # posterior
    P = (1 - K) * P  # posterior variance
    return gaussian(x, P)


def predict_primer(posterior, movement):
    x, P = posterior  # mean and variance of posterior
    dx, Q = movement  # mean and variance of movement
    x = x + dx
    P = P + Q
    return gaussian(x, P)

"""
                                KAlman Filter LOGIC
Initialization

1. Initialize the state of the filter
2. Initialize our belief in the state

Predict

1. Use system behavior to predict state at the next time step
2. Adjust belief to account for the uncertainty in prediction

Update

1. Get a measurement and associated belief about its accuracy
2. Compute residual between estimated state and measurement
3. Compute scaling factor based on whether the measurement
or prediction is more accurate
4. set state between the prediction and measurement based 
on scaling factor
5. update belief in the state based on how certain we are 
in the measurement
"""


# sensor_var = 300.**2
# process_var = 0.001  # 2.
# process_model = gaussian(1., process_var)
# pos = gaussian(0., 500.)
# N = 1000
# dog = DogSimulation(pos.mean, 1., sensor_var, process_var)
# zs = [dog.move_and_sense() for _ in range(N)]
# ps = []
#
# for i in range(N):
#     prior = predict(pos, process_model)
#     pos = update(prior, gaussian(zs[i][1], sensor_var))
#     ps.append(pos.mean)

#book_plots.plot_measurements(zs, lw=1)
###################### test 2  Example: Extreme Amounts of Noise  ##############################################
sensor_var = 20.
process_var = .001
process_model = gaussian(1., process_var)
pos = gaussian(0., 500.)
N = 100
dog = DogSimulation(pos.mean, 1, sensor_var, process_var*10000)
zs, ps = [], []
for _ in range(N):
    dog.velocity += 0.04
    zs.append(dog.move_and_sense())

for z in zs:
    prior = predict(pos, process_model)
    pos = update(prior, gaussian(z[1], sensor_var))
    ps.append(pos.mean)

###################### test 2 ##############################################
###################### test 3 Example: Bad Initial Estimate  ##############################################
sensor_var = 5.**2
process_var = 2.
pos = gaussian(1000., 500.)
process_model = gaussian(1., process_var)
N = 100
dog = DogSimulation(0, 1, sensor_var, process_var)
zs = [dog.move_and_sense() for _ in range(N)]
ps = []

for z in zs:
    prior = predict(pos, process_model)
    pos = update(prior, gaussian(z[1], sensor_var))
    ps.append(pos.mean)
###################### test 3 ##############################################
###################### test 4 Example: Large Noise and Bad Initial Estimate  ##############################################
sensor_var = 30000.
process_var = 2.
pos = gaussian(1000., 500.)
process_model = gaussian(1., process_var)

N = 1000
dog = DogSimulation(0, 1, sensor_var, process_var)
zs = [dog.move_and_sense() for _ in range(N)]
ps = []

for z in zs:
    prior = predict(pos, process_model)
    pos = update(prior, gaussian(z[1], sensor_var))
    ps.append(pos.mean)


###################### test 4 ##############################################
###################### test 5 ##############################################
""" Finally, let's implement the suggestion of using the first measurement as the initial position."""
sensor_var = 30000.
process_var = 2
process_model = gaussian(1., process_var)
N = 1000
dog = DogSimulation(0, 1, sensor_var, process_var)
zs = [dog.move_and_sense() for _ in range(N)]

pos = gaussian(zs[0][1], 500.)
ps = []
for z in zs:
    prior = predict(pos, process_model)
    pos = update(prior, gaussian(z[1], sensor_var))
    ps.append(pos.mean)
###################### test 5 ##############################################




plt.legend(loc=4)
fig, ax = plt.subplots()
zzs=np.array(zs)
x,y=zzs.T
plt.scatter(x,y,color=f'red', s=10)
plot_filter(ps)
plt.grid()
plt.show()

