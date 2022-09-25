################  Steering the Robot ################## turning real word movements


import numpy as np


from a10_5_UKF_ROBOT_LOCALISATION import run_localization

landmarks = np.array([[5, 10], [10, 5], [15, 15], [20, 5],
                      [0, 30], [50, 30], [40, 10]])
dt = 0.1
wheelbase = 0.5
sigma_range = 0.3
sigma_bearing = 0.1


def turn(v, t0, t1, steps):
    return [[v, a] for a in np.linspace(
        np.radians(t0), np.radians(t1), steps)]


# accelerate from a stop
cmds = [[v, .0] for v in np.linspace(0.001, 1.1, 30)]
cmds.extend([cmds[-1]] * 5)

# turn left
v = cmds[-1][0]
cmds.extend(turn(v, 0, 2, 15))
cmds.extend([cmds[-1]] * 5)

# turn right
cmds.extend(turn(v, 2, -2, 15))
cmds.extend([cmds[-1]] * 5)

cmds.extend(turn(v, -2, 0, 15))
cmds.extend([cmds[-1]] * 5)

cmds.extend(turn(v, 0, 1, 25))
cmds.extend([cmds[-1]] * 4)
# ################ run the code #################

ukf = run_localization(
    cmds, landmarks, sigma_vel=0.1, sigma_steer=np.radians(1),
    sigma_range=0.3, sigma_bearing=0.1, step=1,
    ellipse_step=20)
print('final covariance', ukf.P.diagonal())


# ukf = run_localization(
#     cmds, landmarks[0:2], sigma_vel=0.1, sigma_steer=np.radians(1),
#     sigma_range=0.3, sigma_bearing=0.1, step=1,
#     ellipse_step=20)
# print('final covariance', ukf.P.diagonal())