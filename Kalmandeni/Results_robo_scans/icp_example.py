import json
import math

import numpy as np
import matplotlib.pyplot as plt


# icp_known_corresp: performs icp given that the input datasets
# are aligned so that Line1(:, QInd(k)) corresponds to Line2(:, PInd(k))
from Results_robo_scans.transofmation_vector_ukf import DeniTransformation


def coordinates_for_x_y__from_distance_and_angle( orientation_angle, x_base, y_base, angle, distance):
    """
    :param orientation_angle: which direction in the world is oriented the robot
    :param x_base: where is lidar on x
    :param y_base: where is lidar on y
    :param angle: in degree ,0 is +y
    :param distance: from lidar to the point
    :return: new coordiantes of the found point
    """
    new_x = distance * math.sin(math.radians(angle + orientation_angle))
    new_y = distance * math.cos(math.radians(angle + orientation_angle))
    final_x = x_base + new_x
    final_y = y_base + new_y
    final_x = round(final_x, 2)
    final_y = round(final_y, 2)
    if final_x == 0 or final_x == -0:
        final_x = 0
    if final_y == 0 or final_y == -0:
        final_y = 0
    return final_x, final_y



def icp_known_corresp(Line1, Line2, QInd, PInd):
    Q = Line1[:, QInd]
    P = Line2[:, PInd]
    MuQ = compute_mean(Q)
    MuP = compute_mean(P)
    W = compute_W(Q, P, MuQ, MuP)
    [R, t] = compute_R_t(W, MuQ, MuP)
    NewLine = R @ P
    NewLine[0, :] += t[0]
    NewLine[1, :] += t[1]
    E = compute_error(Q, NewLine)
    return [NewLine, E]


# compute_W: compute matrix W to use in SVD
def compute_W(Q, P, MuQ, MuP):
    Q[0, :] -= MuQ[0]
    Q[1, :] -= MuQ[1]
    P[0, :] -= MuP[0]
    P[1, :] -= MuP[1]
    return Q @ P.T


# compute_R_t: compute rotation matrix and translation vector
# based on the SVD as presented in the lecture
def compute_R_t(W, MuQ, MuP):
    U, S, V= np.linalg.svd(W)
    R = V @ U.T
    t = MuQ - (R @ MuP)

    return [R, t]


# compute_mean: compute mean value for a [M x N] matrix
def compute_mean(M):
    # center of mass
    res=np.mean(M, axis=1)

    return res


# compute_error: compute the icp error
def compute_error(Q, OptimizedPoints):
    E = Q - OptimizedPoints
    return np.sqrt(np.sum(E ** 2))


# simply show the two lines
def show_figure(Line1, Line2):
    plt.figure()
    plt.scatter(Line1[0], Line1[1], marker='o', s=2, label='Line 1')
    plt.scatter(Line2[0], Line2[1], s=1, label='Line 2')

    world=400 #400

    plt.xlim([-world, world])
    plt.ylim([-world, world])
    plt.legend()

    plt.show()


# initialize figure
def init_figure():
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()

    line1_fig = plt.scatter([], [], marker='o', s=2, label='Line 1')
    line2_fig = plt.scatter([], [], marker='o', s=1, label='Line 2')
    # plt.title(title)
    plt.xlim([-8, 8])
    plt.ylim([-8, 8])
    plt.legend()

    return fig, line1_fig, line2_fig


# update_figure: show the current state of the lines
def update_figure(fig, line1_fig, line2_fig, Line1, Line2, hold=False):
    line1_fig.set_offsets(Line1.T)
    line2_fig.set_offsets(Line2.T)
    if hold:
        plt.show()
    else:
        fig.canvas.flush_events()
        fig.canvas.draw()
        plt.pause(0.5)


#Data = np.load('icp_data.npz')
# Line1 = Data['LineGroundTruth']
# Line2 = Data['LineMovedCorresp']

################### TEST HERE ###################
with open("lidar_test_UKF.json",'r') as jso:
    testt=json.load(jso)



# a=np.array(testt['1']["node_points"][:166])
# b=np.array(testt['2']["node_points"][:166])
robot_coordinates=[(0,0),(0,10),(0,30),(0,100),(0,100)]
c=1
c_2=1
min_=min(len(testt[f"{c}"]),len(testt[f"{c_2}"]))
start_=0

dt=DeniTransformation()



a=np.array([coordinates_for_x_y__from_distance_and_angle(0,robot_coordinates[c-1][0],robot_coordinates[c-1][1],ang,dist)   for ang,dist in testt[f"{c}"][start_:min_]])
b=np.array([coordinates_for_x_y__from_distance_and_angle(0,robot_coordinates[c_2-1][0],robot_coordinates[c_2-1][1],ang,dist)   for ang,dist in testt[f"{c_2}"][start_:min_]])
#b=np.array(dt.translocation(b,100,100,190))
Line1=np.array([a[:,0],a[:,1]])
Line2=np.array([b[:,0],b[:,1]])
# Line1=np.array([[0,2,4],[0,2,4]],dtype='float64')
# Line2=np.array([[-2,0,2],[-2,0,2]],dtype='float64')


# Show the initial positions of the lines
#show_figure(Line1, Line2)




# Perform icp given the correspondences
# for _ in range(2):
#     QInd = np.arange(len(Line1[0])) # are 1 to 1 correspondences for this data
#     PInd = np.arange(len(Line2[0]))
#     [Line2, E] = icp_known_corresp(Line1, Line2, QInd, PInd)
#
# # Show the adjusted positions of the lines
# show_figure(Line1, Line2)
#
# # print the error
# print('Error value is: ', E) # 290 is good

