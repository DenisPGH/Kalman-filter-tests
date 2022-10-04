import json
import math
import numpy as np
import matplotlib.pyplot as plt

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


def center_of_gravity(points):
    points=np.array(points)
    x = points[:,0]
    y = points[:,1]
    centroid = (sum(x) / len(points), sum(y) / len(points))
    return centroid
test_path=[(0,0),(0,10),(0,20),(0,70),(90,0)]
robot_coordinates=[(0,0),(0,10),(0,30),(0,100),(0,100)]

DT=DeniTransformation()

with open("lidar_test_UKF_2.json",'r') as jso:
    land_all_points=json.load(jso)

list_with_coordiantes=[]
dict_node_coord={}
list_with_corrected_coordinates=[]

counter=0
center=0,0
for node,info in land_all_points.items():
    if counter>1:
        break
    cur_list = []
    new_places_=[]
    for angle,dist in info:
        x,y=coordinates_for_x_y__from_distance_and_angle(
            test_path[counter][0],robot_coordinates[counter][0],robot_coordinates[counter][1],angle,dist)
        list_with_coordiantes.append((x,y))
        cur_list.append((x,y))
    dict_node_coord[counter]=cur_list
    center_2 = center_of_gravity(cur_list)
    print(f'{counter} center bevor {center_2}')

    if counter==0:
        center = center_of_gravity(dict_node_coord[counter])
        new_places_=cur_list
    print(f'first center {center}')
    if counter !=0:
        to_x=center_2[0]+center[0]
        to_y=center_2[1]+center[1]
        new_places_ = DT.translocation(cur_list, to_x,to_y, 0)
    center_ = center_of_gravity(new_places_)
    print(f'{counter} center after {center_}')
    list_with_corrected_coordinates.extend(new_places_)

    counter+=1


landmarks=np.array(list_with_coordiantes)
landmarks_2=np.array(list_with_corrected_coordinates)
rob=np.array(robot_coordinates)
## geometric center






# show

plt.scatter(landmarks[:, 1], landmarks[:, 0], s=10,c='green',label=f'landmarks {len(landmarks)}')
plt.scatter(landmarks_2[:, 1], landmarks_2[:, 0], s=10,c='blue',label=f'correct')
plt.scatter(rob[:, 1], rob[:, 0],c='black',label='coordinates robot') # all detected points
plt.title("Translocation tests")
plt.legend(loc='upper left')
plt.grid()
plt.show()
