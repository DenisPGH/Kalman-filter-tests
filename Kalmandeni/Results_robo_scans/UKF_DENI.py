import json
import math
from math import tan, sin, cos, sqrt, atan2
from filterpy.kalman import MerweScaledSigmaPoints
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.stats import plot_covariance_ellipse




def move_steering(x, dt, u, wheelbase):
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
    #print(f"vel= {u[1]}")
    steering_angle = u[1]
    dist = vel * dt

    if abs(steering_angle) > 0.001: # is robot turning?
        beta = (dist / wheelbase) * tan(steering_angle)
        r = wheelbase / tan(steering_angle) # radius

        sinh, sinhb = sin(hdg), sin(hdg + beta)
        cosh, coshb = cos(hdg), cos(hdg + beta)
        return x + np.array([-r*sinh + r*sinhb, r*cosh - r*coshb, beta])
    else: # moving in straight line
        return x + np.array([dist*cos(hdg), dist*sin(hdg), 0])




def move(x, dt, u, wheelbase):
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

    hdg = x[2] # theta
    vel = u[0]
    steering_angle = u[1]
    dist = vel * dt

    if abs(steering_angle) > 0.001: # is robot turning?
        #beta = (dist / wheelbase) * tan(steering_angle)
        beta = steering_angle
        r = wheelbase / tan(steering_angle) # radius


        sinh, sinhb = sin(hdg), sin(hdg )
        cosh, coshb = cos(hdg), cos(hdg)
        #return x + np.array([-r*sinh + r*sinhb, r*cosh - r*coshb, beta])
        #return x + np.array([dist * cos(hdg), dist * sin(hdg), hdg])
        return x + np.array([dist * cos(steering_angle), dist * sin(steering_angle), steering_angle-hdg])

    else: # moving in straight line
        #print(x + np.array([dist*cos(hdg), dist*sin(hdg), 0]))
        return x + np.array([dist*cos(hdg), dist*sin(hdg), 0])



def normalize_angle_steering(x):
    """
    x: shoud be in radians
    this funktion handle the different 360-1 degre"""
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    return x


def normalize_angle(x):
    """
    x: shoud be in radians
    this funktion handle the different 360-1 degre"""
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    return x

def residual_h(a, b):
    """The state vector has the bearing at index 2, but the measurement vector has it at index 1,
     so we need to write functions to handle each."""
    y = a - b
    # data in format [dist_1, bearing_1, dist_2, bearing_2,...]
    for i in range(0, len(y), 2):
        y[i + 1] = normalize_angle(y[i + 1])
    return y

def residual_x(a, b):
    """The state vector has the bearing at index 2, but the measurement vector has it at index 1,
     so we need to write functions to handle each."""
    y = a - b
    y[2] = normalize_angle(y[2])
    return y

def Hx(x, landmarks):
    """
    x:
    landmarks:
    return : [dist_to_1, bearing_to_1, dist_to_2, bearing_to_2, ...].
     takes a state variable and returns the measurement
    that would correspond to that state. """
    hx = []
    for lmark in landmarks:
        px, py = lmark
        dist = sqrt((px - x[0])**2 + (py - x[1])**2)
        angle = atan2(py - x[1], px - x[0])
        hx.extend([dist, normalize_angle(angle - x[2])])
    return np.array(hx)

def state_mean(sigmas, Wm):
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

def z_mean(sigmas, Wm):
    """

    :param sigmas:
    :param Wm:
    :return:
    """
    z_count = sigmas.shape[1]
    x = np.zeros(z_count)

    for z in range(0, z_count, 2):
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, z+1]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, z+1]), Wm))

        x[z] = np.sum(np.dot(sigmas[:,z], Wm))
        x[z+1] = atan2(sum_sin, sum_cos)
    return x



#################### Implementation ###########################################



def run_localization(
        cmds, landmarks, sigma_vel, sigma_steer, sigma_range,
        sigma_bearing, ellipse_step=1, step=10):
    """

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
    plt.figure()
    points = MerweScaledSigmaPoints(n=3, alpha=.00001, beta=2, kappa=0,
                                    subtract=residual_x)
    ukf = UKF(dim_x=3, dim_z=2 * len(landmarks), fx=move, hx=Hx,
              dt=dt, points=points, x_mean_fn=state_mean,
              z_mean_fn=z_mean, residual_x=residual_x,
              residual_z=residual_h)

    ukf.x = np.array([0, 0, 0]) # [2, 6, .3] # here is the start position and orientation
    ukf.P = np.diag([.1, .1, .05])
    ukf.R = np.diag([sigma_range ** 2,
                     sigma_bearing ** 2] * len(landmarks))
    ukf.Q = np.eye(3) * 0.0001

    sim_pos = ukf.x.copy()

    # plot landmarks
    if len(landmarks) > 0:
        #plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='s', s=60)
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=30)

    track = []
    for i, u in enumerate(cmds):
        sim_pos = move(sim_pos, dt, u, wheelbase)
        #print(sim_pos)
        track.append(sim_pos)

        if i % step == 0:
            ukf.predict(u=u, wheelbase=wheelbase)

            if i % ellipse_step == 0:
                plot_covariance_ellipse(
                    (ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=6,
                    facecolor='k', alpha=0.3)

            x, y = sim_pos[0], sim_pos[1]
            z = []
            for lmark in landmarks:
                dx, dy = lmark[0] - x, lmark[1] - y
                d = sqrt(dx ** 2 + dy ** 2) + randn() * sigma_range
                bearing = atan2(lmark[1] - y, lmark[0] - x)
                a = (normalize_angle(bearing - sim_pos[2] +
                                     randn() * sigma_bearing))
                z.extend([d, a])
            ukf.update(z, landmarks=landmarks)
            #print(f"x: {ukf.x[0]} , y: {ukf.x[1]} , Theta: {math.degrees(ukf.x[2])}")
            if i % ellipse_step == 0:
                plot_covariance_ellipse(
                    (ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=6,
                    facecolor='g', alpha=0.8)


    track = np.array(track)
    plt.plot(track[:, 0], track[:, 1], color='k', lw=2)
    plt.axis('equal')
    plt.title("UKF Robot localization")
    ###### show orientation robot ############
    #plt.plot(ukf.x[0], ukf.x[1], marker=(5, 0, math.degrees(ukf.x[2])), markersize=20, linestyle='None',color='red')
    #plt.plot(ukf.x[0], ukf.x[1],color='blue',lw=20)
    #print(f"x: {ukf.x[0]} , y: {ukf.x[1]} , Theta: {math.degrees(ukf.x[2])}")
    plt.grid()
    plt.show()

    return ukf


############### run the code for the moving ##########################
# with open("js.json",'r') as jso:
#     points=json.load(jso)
# landmarks=[l['node_points'] for l in points.values()]
# landmarks=np.array([[y,x] for x,y in landmarks[0]])
# #landmarks=np.array(landmarks[0])

landmarks = np.array([[5, 10], [10, 5], [15, 15], [20, 5], [0, 30], [50, 30], [40, 10]])
dt = 1
wheelbase = 0.5
sigma_range = 0.3
sigma_bearing = 0.1

def turn(v, t0, t1, steps):
    """

    :param v: velocity
    :param t0:
    :param t1:
    :param steps:
    :return:
    """

    return [[v, a] for a in np.linspace(
        np.radians(t0), np.radians(t1), steps)]

def pi_to_pi(x):
    # x in radians
    x=x%360
    return x




# accelerate from a stop
#cmds = [[v, .0] for v in np.linspace(0.001, 5, 3)] #(,velosity,)
# cmds.extend([cmds[-1]] * 50)
#print(cmds)

# turn left
# v = cmds[-1][0]
# print(v) # velocity
# v=5
# cmds=[]
# cmds.extend(turn(v, 0, 5, 60))
# cmds.extend([cmds[-1]] * 3)
# for a in cmds:
#     print(a)  # a[0]= velociti   , a[1] =
#
# # turn right
# cmds.extend(turn(v, 2, -2, 15))
# cmds.extend([cmds[-1]] * 200)
#
# cmds.extend(turn(v, -2, 0, 15))
# cmds.extend([cmds[-1]] * 150)
#
# cmds.extend(turn(v, 0, 1, 25))
# cmds.extend([cmds[-1]] * 100)
# ################ run the code #################

dist=100

ang=315
# if not 0<=ang<=180:
#     velocity=dist
#     angle_degrees=ang
# else:
velocity=1 #cm/sec ili dist
angle_degrees=ang
angle_radians=math.radians(angle_degrees)
cmds= [[velocity,angle_radians] if x<15  else [velocity,angle_radians] for x in range(dist) ]


ukf = run_localization(
    cmds, landmarks, sigma_vel=0.1, sigma_steer=np.radians(1),
    sigma_range=0.3, sigma_bearing=0.1, step=1,
    ellipse_step=10)
#print('final covariance', ukf.P.diagonal())
#orientir=math.degrees(ukf.x[2])
orientir=pi_to_pi(math.degrees(ukf.x[2]))


print(f"x: {ukf.x[0]} , y: {ukf.x[1]} , Theta_-pi: {math.degrees(ukf.x[2])}")
print(f"x: {ukf.x[0]} , y: {ukf.x[1]} , Theta_2pi: {orientir}")

### show car orientation
# plt.plot(ukf.x[0], ukf.x[1], marker=(5, 0, math.degrees(ukf.x[2])), markersize=20, linestyle='None',color='red')
# plt.show()

a=normalize_angle(angle_radians)
print(math.degrees(a))





