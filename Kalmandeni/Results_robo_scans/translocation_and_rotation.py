import json
import math
import numpy as np
import matplotlib.pyplot as plt

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

test_path=[(0,0),(0,10),(0,20),(0,70),(90,0)]
robot_coordinates=[(0,0),(0,10),(0,30),(0,100),(0,100)]
with open("lidar_test_UKF_2.json",'r') as jso:
    land_all_points=json.load(jso)

list_with_coordiantes=[]
counter=0
for node,info in land_all_points.items():
    if counter>4:
        break

    for angle,dist in info:
        x,y=coordinates_for_x_y__from_distance_and_angle(
            test_path[counter][0],robot_coordinates[counter][0],robot_coordinates[counter][1],angle,dist)
        list_with_coordiantes.append((x,y))
    counter+=1


landmarks=np.array(list_with_coordiantes)



# show

plt.scatter(landmarks[:, 1], landmarks[:, 0], s=20,c='green',label=f'landmarks {len(landmarks)}')
plt.title("Translocation tests")
plt.legend(loc='upper left')
plt.grid()
plt.show()
