import json
import  numpy as np

from Results_robo_scans.ICP_other.kd_trees import kdtree
from Results_robo_scans.icp_example import coordinates_for_x_y__from_distance_and_angle

with open("lidar_test_UKF_2.json",'r') as jso:
    testt=json.load(jso)

# print(np.array(testt['1'])[:,1])

robot_coordinates=[(0,0,0),(0,10,0),(0,30,0),(0,100,0),(0,100,90)] # (x,y,theta)
c=1
c_2=4
min_=min(len(testt[f"{c}"]),len(testt[f"{c_2}"]))
start_=0

aa=[coordinates_for_x_y__from_distance_and_angle(robot_coordinates[c-1][2],robot_coordinates[c-1][0],robot_coordinates[c-1][1],ang,dist)   for ang,dist in testt[f"{c}"][start_:min_]]

print(len(aa))

m=kdtree(aa)
print(m)