import json

from Results_robo_scans.icp_example import coordinates_for_x_y__from_distance_and_angle
from Results_robo_scans.icp_git.basicICP import icp_point_to_plane_lm, icp_point_to_plane
import numpy as np
import matplotlib.pyplot as plt
from Results_robo_scans.transofmation_vector_ukf import DeniTransformation
# fileOriginal = 'original.xyz'
# deformed = 'deformed.xyz'
#
# source_points = read_file_original(fileOriginal)
# dest_points_et_normal = read_file_deformed(deformed)

def icp_point_to_point_lm(source_points, dest_points, initial, loop):
    """
    Point to point matching using Gauss-Newton

    source_points:  nx3 matrix of n 3D points

    dest_points: nx3 matrix of n 3D points, which have been obtained by some rigid deformation
    of 'source_points'

    initial: 1x6 matrix, denoting alpha, beta, gamma (the Euler angles for rotation and tx, ty, tz
    (the translation along three axis). this is the initial estimate of the transformation
    between 'source_points' and 'dest_points'

    loop: start with zero, to keep track of the number of times it loops, just a very crude way to
     control the recursion

    """
    J = []
    e = []
    for i in range(0, dest_points.shape[0] - 1):
        dx = dest_points[i][0]
        dy = dest_points[i][1]
        dz = dest_points[i][2]
        sx = source_points[i][0]
        sy = source_points[i][1]
        sz = source_points[i][2]
        alpha = initial[0][0]
        beta = initial[1][0]
        gamma = initial[2][0]
        tx = initial[3][0]
        ty = initial[4][0]
        tz = initial[5][0]
        a1 = (-2 * beta * sx * sy) - (2 * gamma * sx * sz) + (2 * alpha * ((sy * sy) + (sz * sz))) + (
                    2 * ((sz * dy) - (sy * dz))) + 2 * ((sy * tz) - (sz * ty))
        a2 = (-2 * alpha * sx * sy) - (2 * gamma * sy * sz) + (2 * beta * ((sx * sx) + (sz * sz))) + (
                    2 * ((sx * dz) - (sz * dx))) + 2 * ((sz * tx) - (sx * tz))
        a3 = (-2 * alpha * sx * sz) - (2 * beta * sy * sz) + (2 * gamma * ((sx * sx) + (sy * sy))) + (
                    2 * ((sy * dx) - (sx * dy))) + 2 * ((sx * ty) - (sy * tx))
        a4 = 2 * (sx - (gamma * sy) + (beta * sz) + tx - dx)
        a5 = 2 * (sy - (alpha * sz) + (gamma * sx) + ty - dy)
        a6 = 2 * (sz - (beta * sx) + (alpha * sy) + tz - dz)
        _residual = (a4 * a4 / 4) + (a5 * a5 / 4) + (a6 * a6 / 4)
        _J = np.array([a1, a2, a3, a4, a5, a6])
        _e = np.array([_residual])  # error
        J.append(_J)
        e.append(_e)
    jacobian = np.array(J)
    residual = np.array(e)
    update = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(jacobian), jacobian)), np.transpose(jacobian)), residual)

    initial = initial + update
    print(initial)
    loop = loop + 1
    if (loop < 3):
        icp_point_to_point_lm(source_points, dest_points, initial, loop)


with open("lidar_test_UKF_2.json",'r') as jso:
    testt=json.load(jso)


robot_coordinates=[(0,0,0),(0,10,0),(0,30,0),(0,100,0),(0,100,90)] # (x,y,theta)
c=1
c_2=4
min_=min(len(testt[f"{c}"]),len(testt[f"{c_2}"]))
start_=0

# a =np.array(testt['1'])
# b=np.array(testt['2'])
aa=[coordinates_for_x_y__from_distance_and_angle(robot_coordinates[c-1][2],robot_coordinates[c-1][0],robot_coordinates[c-1][1],ang,dist)   for ang,dist in testt[f"{c}"][start_:min_]]
bb=[coordinates_for_x_y__from_distance_and_angle(robot_coordinates[c_2-1][2],robot_coordinates[c_2-1][0],robot_coordinates[c_2-1][1],ang,dist)   for ang,dist in testt[f"{c_2}"][start_:min_]]
#b=np.array(dt.translocation(b,100,100,190))
aa.append([robot_coordinates[c-1][0],robot_coordinates[c-1][1]])
bb.append([robot_coordinates[c_2-1][0],robot_coordinates[c_2-1][1]])
a=np.array(aa)
b=np.array(bb)

a=np.insert(a, 2, 0, axis=1) # add a Z axis=0

b=np.insert(b, 2, 0, axis=1) # add a Z axis=0
b=np.insert(b, 3, 0, axis=1) # add a Z axis=0
b=np.insert(b, 4, 0, axis=1) # add a Z axis=0
b=np.insert(b, 5, 0, axis=1) # add a Z axis=0
#print(b)
# b=np.insert(b, 3, 0, axis=0) # add a Z axis=0
# b=np.insert(b, 4, 0, axis=1) # add a Z axis=0
# b=np.insert(b, 5, 0, axis=1) # add a Z axis=0

Line1=np.array([a[:,0],a[:,1],a[:,2]])
#Line2=np.array([b[:,0],b[:,1],b[:,2]])
Line2=np.array([b[:,0],b[:,1],b[:,2],b[:,3],b[:,4],b[:,5]])
test_=Line2.copy()
# a =np.array([np.array([x,y,1,1,1,1])for x,y in testt['1']["node_points"]])
# b =np.array([np.array([x,y,1,1,1,1])for x,y in testt['5']["node_points"]])
# #b=np.array(testt['5']["node_points"])
# Line1=a
# Line2=b
# print(Line1)

#initial = np.array([[0.01], [0.05], [0.01], [0.001], [0.001], [0.001]])
calc_dist_y= robot_coordinates[c_2 - 1][1] - robot_coordinates[c - 1][1]
print('dddd', calc_dist_y)
initial = np.array([[0], [0], [0], [0], [0], [0]]) # alpha, beta, gamma, tx, ty, tz




transpose=icp_point_to_plane(Line2,Line1,0)
#print(transpose)
print("dddddddddddddddddddddddddddddddd")

icp_point_to_point_lm(Line2,Line1,initial,0)
print("dddddddddddddddddddddddddddddddd")
#icp_point_to_plane_lm(Line2,Line1,initial,0)

#################

a=np.delete(a,2,axis=1) # remove Z axis

b=np.delete(b,5,axis=1) # remove Z axis
b=np.delete(b,4,axis=1) # remove Z axis
b=np.delete(b,3,axis=1) # remove Z axis
b=np.delete(b,2,axis=1) # remove Z axis



DT=DeniTransformation()
c=DT.translocation_new_version(b,7,25.53,2)
#c=[coordinates_for_x_y__from_distance_and_angle(2.41,3.95,126.98,ang,dist)   for ang,dist in testt[f"{c_2}"][start_:min_]]
#b=np.array(dt.translocation(b,100,100,190))
c=np.array(c)
# x,y=data.T
# x_t,y_t=data_transf.T
# x_w,y_w=world_frame.T
res=np.array(np.dot(test_.T,transpose))
# res=np.delete(res,5,axis=1) # remove Z axis
# res=np.delete(res,4,axis=1) # remove Z axis
# res=np.delete(res,3,axis=1) # remove Z axis
# res=np.delete(res,2,axis=1) # remove Z axis
#print(res)
robot=np.array(robot_coordinates)
robot=np.delete(robot,2,axis=1) # remove Z axis

plt.scatter(a[:,0], a[:,1], color=f'red', s=5,label='first scan')
plt.scatter(b[:,0], b[:,1], color=f'green', s=5,label='second scan')
plt.scatter(c[:,0], c[:,1], color=f'blue', s=5,label='transformed')
#plt.scatter(res[:,0], res[:,1], color=f'yellow', s=5,label='transformed_icp')
plt.plot(robot[:,0], robot[:,1], color=f'black',label='robot')
plt.legend(loc='lower right')
plt.grid()
plt.show()


