import json

from Results_robo_scans.icp_example import coordinates_for_x_y__from_distance_and_angle
from Results_robo_scans.icp_git.basicICP import icp_point_to_plane_lm, icp_point_to_plane, icp_point_to_point_lm
import numpy as np
import matplotlib.pyplot as plt
from Results_robo_scans.transofmation_vector_ukf import DeniTransformation
# fileOriginal = 'original.xyz'
# deformed = 'deformed.xyz'
#
# source_points = read_file_original(fileOriginal)
# dest_points_et_normal = read_file_deformed(deformed)


with open("lidar_test_UKF.json",'r') as jso:
    testt=json.load(jso)

robot_coordinates=[(0,0),(0,10),(0,30),(0,100),(90,100)]
c=1
c_2=4
min_=min(len(testt[f"{c}"]),len(testt[f"{c_2}"]))
start_=0

# a =np.array(testt['1'])
# b=np.array(testt['2'])
a=np.array([coordinates_for_x_y__from_distance_and_angle(0,robot_coordinates[c-1][0],robot_coordinates[c-1][1],ang,dist)   for ang,dist in testt[f"{c}"][start_:min_]])
b=np.array([coordinates_for_x_y__from_distance_and_angle(0,robot_coordinates[c_2-1][0],robot_coordinates[c_2-1][1],ang,dist)   for ang,dist in testt[f"{c_2}"][start_:min_]])
#b=np.array(dt.translocation(b,100,100,190))

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
initial = np.array([[0], [0], [0], [0], [0], [0]]) # start position




transpose=icp_point_to_plane(Line2,Line1,0)
#print(transpose)
print("dddddddddddddddddddddddddddddddd")

#icp_point_to_point_lm(Line2,Line1,initial,0)
print("dddddddddddddddddddddddddddddddd")
#icp_point_to_plane_lm(Line2,Line1,initial,0)

#################

a=np.delete(a,2,axis=1) # remove Z axis

b=np.delete(b,5,axis=1) # remove Z axis
b=np.delete(b,4,axis=1) # remove Z axis
b=np.delete(b,3,axis=1) # remove Z axis
b=np.delete(b,2,axis=1) # remove Z axis



DT=DeniTransformation()
c=DT.translocation(b,0,0,0)
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

plt.scatter(a[:,0], a[:,1], color=f'red', s=5,label='first scan')
plt.scatter(b[:,0], b[:,1], color=f'green', s=5,label='second scan')
#plt.scatter(c[:,0], c[:,1], color=f'blue', s=5,label='transformed')
#plt.scatter(res[:,0], res[:,1], color=f'yellow', s=5,label='transformed_icp')
plt.legend(loc='upper left')
plt.grid()
plt.show()


