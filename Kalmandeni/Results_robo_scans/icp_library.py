import json

from simpleicp import PointCloud, SimpleICP # pip unistall simpleicp
import numpy as np
# robot
from Results_robo_scans.icp_example import coordinates_for_x_y__from_distance_and_angle

with open("lidar_test_UKF_2.json",'r') as jso:
    testt=json.load(jso)

robot_coordinates=[(0,0,0),(0,10,0),(0,30,0),(0,100,0),(0,100,90)] # (x,y,theta)
c=1
c_2=1
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

# Read point clouds from xyz files into n-by-3 numpy arrays
# X_fix = np.genfromtxt("bunny_part1.xyz")
# X_mov = np.genfromtxt("bunny_part2.xyz")
X_fix=a[:200]
X_mov=b[:200]

# Create point cloud objects
pc_fix = PointCloud(X_fix, columns=["x", "y", "z"])
pc_mov = PointCloud(X_mov, columns=["x", "y", "z"])
print(pc_fix)

# pc_fix = PointCloud(X_fix, columns=[0,1,2])
# pc_mov = PointCloud(X_mov, columns=[0,1,2])

# Create simpleICP object, add point clouds, and run algorithm!
icp = SimpleICP()
icp.add_point_clouds(pc_fix, pc_mov)
H, X_mov_transformed, rigid_body_transformation_params = icp.run(max_overlap_distance=1)