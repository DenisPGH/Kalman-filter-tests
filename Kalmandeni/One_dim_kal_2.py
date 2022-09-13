
import numpy as np
import copy
import math
from numpy.random import randn
from collections import namedtuple

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
process_var = 1.  # variance in the dog's movement
sensor_var = 2.  # variance in the sensor
x = gaussian(0., 20. ** 2)  # dog's position, N(0, 20**2)
velocity = 1
dt = 1.  # time step in seconds
process_model = gaussian(velocity * dt, process_var)  # displacement to add to x

# simulate dog and get measurements
dog = DogSimulation(
    x0=x.mean,
    velocity=process_model.mean,
    measurement_var=sensor_var,
    process_var=process_model.var)

# create list of measurements
zs = [dog.move_and_sense() for _ in range(10)]
print('PREDICT\t\t\tUPDATE')
print('     x      var\t\t  z\t    x      var')

# perform Kalman filter on measurement z
for z in zs:
    prior = predict(x, process_model)
    likelihood = gaussian(z[1], sensor_var)
    x = update(prior, likelihood)
    print(f"{prior},  {z[1]},    {x},")


print(f'final estimate:        {x.mean:10.3f}')
print(f'actual final position: {dog.x:10.3f}')

