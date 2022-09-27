import json
import math
import time
from math import tan, sin, cos, sqrt, atan2
from filterpy.kalman import MerweScaledSigmaPoints
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.stats import plot_covariance_ellipse


class UKFDeni:
    def __init__(self):
        self.dt=1
        self.wheelbase=0.18
        self.end_x=0
        self.end_y=0
        self.end_theta=0
        self.start_x = 0
        self.start_y = 0
        self.start_theta = 0
        self.track=[]
        self.VELOCITY=1 # cm/sec ili dist

    def pi_to_pi(self,x):
        # x in radians
        """
        convert -180,180 to 0-360

        :param x: angle in radians
        :return:
        """
        x = x % 360
        return x


    def move_steering(self,x, dt, u, wheelbase):
        """
        this is for =>  state transition function f(x).
        imitate the moving of the robot
        :param x: position [x,y]
        :param dt: time interval
        :param u: control motion  u=[v,alpha].T
        :param wheelbase: how wide is the robot base
        :return: robot goes ahead or turning
        """

        hdg = x[2]
        vel = u[0]
        # print(f"vel= {u[1]}")
        steering_angle = u[1]
        dist = vel * dt

        if abs(steering_angle) > 0.001:  # is robot turning?
            beta = (dist / wheelbase) * tan(steering_angle)
            r = wheelbase / tan(steering_angle)  # radius

            sinh, sinhb = sin(hdg), sin(hdg + beta)
            cosh, coshb = cos(hdg), cos(hdg + beta)
            return x + np.array([-r * sinh + r * sinhb, r * cosh - r * coshb, beta])
        else:  # moving in straight line
            return x + np.array([dist * cos(hdg), dist * sin(hdg), 0])

    def move(self,x, dt, u, wheelbase):
        """
        move rotation my car
        this is for =>  state transition function f(x).
        imitate the moving of the robot
        :param x: position [x,y,Theta]
        :param dt: time interval
        :param u: control motion  u=[velcity,alpha].T
        :param wheelbase: how wide is the robot base
        :return: robot goes ahead or turning
        """

        hdg = x[2]  # theta
        vel = u[0]
        steering_angle = u[1]
        dist = vel * dt

        if abs(steering_angle) > 0.001:  # is robot turning?
            # beta = (dist / wheelbase) * tan(steering_angle)
            beta = steering_angle
            r = wheelbase / tan(steering_angle)  # radius

            sinh, sinhb = sin(hdg), sin(hdg)
            cosh, coshb = cos(hdg), cos(hdg)
            # return x + np.array([-r*sinh + r*sinhb, r*cosh - r*coshb, beta])
            # return x + np.array([dist * cos(hdg), dist * sin(hdg), hdg])
            return x + np.array([dist * cos(steering_angle), dist * sin(steering_angle), steering_angle - hdg])

        else:  # moving in straight line
            # print(x + np.array([dist*cos(hdg), dist*sin(hdg), 0]))
            return x + np.array([dist * cos(hdg), dist * sin(hdg), 0])

    def normalize_angle_steering(self,x):
        """
        x: shoud be in radians
        this funktion handle the different 360-1 degre"""
        x = x % (2 * np.pi)  # force in range [0, 2 pi)
        if x > np.pi:  # move to [-pi, pi)
            x -= 2 * np.pi
        return x

    def normalize_angle(self,x):
        """
        x: shoud be in radians
        this funktion handle the different 360-1 degre"""
        x = x % (2 * np.pi)  # force in range [0, 2 pi)
        if x > np.pi:  # move to [-pi, pi)
            x -= 2 * np.pi
        return x

    def residual_h(self,a, b):
        """The state vector has the bearing at index 2, but the measurement vector has it at index 1,
         so we need to write functions to handle each."""
        y = a - b
        # data in format [dist_1, bearing_1, dist_2, bearing_2,...]
        for i in range(0, len(y), 2):
            y[i + 1] = self.normalize_angle(y[i + 1])
        return y

    def residual_x(self,a, b):
        """The state vector has the bearing at index 2, but the measurement vector has it at index 1,
         so we need to write functions to handle each."""
        y = a - b
        y[2] = self.normalize_angle(y[2])
        return y

    def Hx(self,x, landmarks):
        """
        x:
        landmarks:
        return : [dist_to_1, bearing_to_1, dist_to_2, bearing_to_2, ...].
         takes a state variable and returns the measurement
        that would correspond to that state. """
        hx = []
        for lmark in landmarks:
            px, py = lmark
            dist = sqrt((px - x[0]) ** 2 + (py - x[1]) ** 2)
            angle = atan2(py - x[1], px - x[0])
            hx.extend([dist, self.normalize_angle(angle - x[2])])
        return np.array(hx)

    def state_mean(self,sigmas, Wm):
        """

        :param sigmas:
        :param Wm:
        :return:
        """

        x = np.zeros(3)

        sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
        x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
        x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
        x[2] = atan2(sum_sin, sum_cos)
        return x

    def z_mean(self,sigmas, Wm):
        """

        :param sigmas:
        :param Wm:
        :return:
        """
        z_count = sigmas.shape[1]
        x = np.zeros(z_count)

        for z in range(0, z_count, 2):
            sum_sin = np.sum(np.dot(np.sin(sigmas[:, z + 1]), Wm))
            sum_cos = np.sum(np.dot(np.cos(sigmas[:, z + 1]), Wm))

            x[z] = np.sum(np.dot(sigmas[:, z], Wm))
            x[z + 1] = atan2(sum_sin, sum_cos)
        return x

    def localization(self,start_x,start_y,start_theta,direction,dist,
            landmarks, sigma_vel, sigma_steer, sigma_range,
            sigma_bearing, ellipse_step=1, step=10):
        """
        :param start_x: start x position
        :param start_y: start y position
        :param start_theta: start orientation 0-360 degrees
        :param direction: which direction to go 0-360 degrees
        :param dist: how far in cm
        :param cmds: [velocity(cm/sec), angle turning (in radians)]
        :param landmarks: [[x,y],[x,y]]
        :param sigma_vel:
        :param sigma_steer:
        :param sigma_range:
        :param sigma_bearing:
        :param ellipse_step:
        :param step:
        :return:
        """

        angle_radians = math.radians(direction)
        cmds = [[self.VELOCITY, angle_radians]] * dist
        points = MerweScaledSigmaPoints(n=3, alpha=.00001, beta=2, kappa=0,
                                        subtract=self.residual_x)
        ukf = UKF(dim_x=3, dim_z=2 * len(landmarks), fx=self.move, hx=self.Hx,
                  dt=self.dt, points=points, x_mean_fn=self.state_mean,
                  z_mean_fn=self.z_mean, residual_x=self.residual_x,
                  residual_z=self.residual_h)

        ukf.x = np.array([start_x, start_y, math.radians(start_theta)])  # [2, 6, .3] # here is the start position and orientation
        ukf.P = np.diag([30, 30, 1.6])
        ukf.R = np.diag([sigma_range ** 2,
                         sigma_bearing ** 2] * len(landmarks))
        ukf.Q = np.eye(3) * 0.01 # 0.0001

        sim_pos = ukf.x.copy()
        #################
        # if len(landmarks) > 0:
        #     plt.scatter(landmarks[:, 1], landmarks[:, 0], s=30)
        self.track = [] # track the current comand of moving
        for i, u in enumerate(cmds):
            sim_pos = self.move(sim_pos, self.dt, u, self.wheelbase)
            self.track.append(sim_pos)
            if i % step == 0:
                ukf.predict(u=u, wheelbase=self.wheelbase)
                if i % ellipse_step == 0:
                    plot_covariance_ellipse(
                        (ukf.x[1], ukf.x[0]), ukf.P[0:2, 0:2], std=6,
                        facecolor='k', alpha=0.3)

                x, y = sim_pos[1], sim_pos[0]
                z = []
                for lmark in landmarks:
                    dx, dy = lmark[1] - x, lmark[0] - y
                    d = sqrt(dx ** 2 + dy ** 2) + randn() * sigma_range
                    bearing = atan2(lmark[0] - y, lmark[1] - x)
                    a = (self.normalize_angle(bearing - sim_pos[2] +
                                         randn() * sigma_bearing))
                    z.extend([d, a])
                ukf.update(z, landmarks=landmarks)
                if i % ellipse_step == 0:
                    plot_covariance_ellipse(
                        (ukf.x[1], ukf.x[0]), ukf.P[0:2, 0:2], std=6,facecolor='g', alpha=0.8)

        self.track = np.array(self.track)
        self.end_x=ukf.x[0]
        self.end_y=ukf.x[1]
        self.end_theta=self.pi_to_pi(math.degrees(ukf.x[2]))
        self.start_x = self.end_x
        self.start_y = self.end_y
        self.start_theta = self.end_theta
        return self.end_x, self.end_y, self.end_theta
