from collections import namedtuple
gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: f'𝒩(μ={s[0]:.3f}, 𝜎²={s[1]:.3f})'

g1 = gaussian(3.4, 10.1)
g2 = gaussian(mean=4.5, var=0.2**2)
print(g1)
print(g2)

def predict(pos, movement):
    return gaussian(pos.mean + movement.mean, pos.var + movement.var)

pos = gaussian(10., .2**2)
move = gaussian(15., .7**2)
res=predict(pos, move)
print(res)
######################################################
print('UPDATE WITH GAUSSIANS')

def gaussian_multiply(g1, g2):
    #print(f"TEst {g1.var}, {g2.mean}, {g2.var} *{g1.mean}")
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)

def update(prior, likelihood):
    posterior = gaussian_multiply(likelihood, prior)
    return posterior

def update_dog(dog_pos, measurement):
    estimated_pos = gaussian_multiply(measurement, dog_pos)
    return estimated_pos

# test the update function
predicted_pos = gaussian(10., .2**2)
measured_pos = gaussian(11., .1**2)
estimated_pos = update(predicted_pos, measured_pos)
print(estimated_pos)

#### Understanding Gaussian Multiplication
import filterpy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

def plot_products(g1, g2):
    plt.figure()
    product = gaussian_multiply(g1, g2)

    xs = np.arange(0, 15, 0.1)
    ys = [stats.gaussian(x, g1.mean, g1.var) for x in xs]
    plt.plot(xs, ys, label='$\mathcal{N}$' + f'$({g1.mean},{g1.var})$')

    ys = [stats.gaussian(x, g2.mean, g2.var) for x in xs]
    plt.plot(xs, ys, label='$\mathcal{N}$' + '$({g2.mean},{ge.var})$')

    ys = [stats.gaussian(x, product.mean, product.var) for x in xs]
    plt.plot(xs, ys, label='product', ls='--')
    plt.legend()
    plt.show()


z1 = gaussian(10.2, 1)
z2 = gaussian(9.7, 1)

z1 = gaussian(10.2, 0.1)
z2 = gaussian(6, 4)

plot_products(z1, z2)

### First Kalman Filter
import copy
import math
from numpy.random import randn
from collections import namedtuple


print('FIRST KALMAN FILTER')
# class DogSimulation(object):
#
#     def __init__(self, x0=0, velocity=1,
#                  measurement_var=0.0, process_var=0.0):
#         """ x0 - initial position
#             velocity - (+=right, -=left)
#             measurement_variance - variance in measurement m^2
#             process_variance - variance in process (m/s)^2
#         """
#         self.x = x0
#         self.velocity = velocity
#         self.measurement_noise = math.sqrt(measurement_var)
#         self.process_noise = math.sqrt(process_var)
#
#
#     def move(self, dt=1.0):
#         '''Compute new position of the dog assuming `dt` seconds have
#         passed since the last update.'''
#         # compute new position based on velocity. Add in some
#         # process noise
#         velocity = self.velocity + randn() * self.process_noise * dt
#         self.x += velocity * dt
#
#
#     def sense_position(self):
#         # simulate measuring the position with noise
#         return self.x + randn() * self.measurement_noise
#
#
#     def move_and_sense(self, dt=1.0):
#         self.move(dt)
#         x = copy.deepcopy(self.x)
#         return x, self.sense_position()
#
#
#     def run_simulation(self, dt=1, count=1):
#         """ simulate the dog moving over a period of time.
#         **Returns**
#         data : np.array[float, float]
#             2D array, first column contains actual position of dog,
#             second column contains the measurement of that position
#         """
#         return np.array([self.move_and_sense(dt) for i in range(count)])
#
#
# np.random.seed(13)
#
# process_var = 1.  # variance in the dog's movement
# sensor_var = 2.  # variance in the sensor
#
# x = gaussian(0., 20. ** 2)  # dog's position, N(0, 20**2)
# velocity = 1
# dt = 1.  # time step in seconds
# process_model = gaussian(velocity * dt, process_var)  # displacement to add to x
#
# # simulate dog and get measurements
# dog = DogSimulation(
#     x0=x.mean,
#     velocity=process_model.mean,
#     measurement_var=sensor_var,
#     process_var=process_model.var)
#
# # create list of measurements
# zs = [dog.move_and_sense() for _ in range(10)]
#
# print(zs)
#
# print('PREDICT\t\t\tUPDATE')
# print('     x      var\t\t  z\t    x      var')
#
# # perform Kalman filter on measurement z
# for z in zs:
#     prior = predict(x, process_model)
#     likelihood = gaussian(z, sensor_var)
#     x = update(prior, likelihood)
#     print(f"{prior}, {x}, {z}")
#
#     #kf_internal.print_gh(prior, x, z)
#
# print()
# print(f'final estimate:        {x.mean:10.3f}')
# print(f'actual final position: {dog.x:10.3f}')