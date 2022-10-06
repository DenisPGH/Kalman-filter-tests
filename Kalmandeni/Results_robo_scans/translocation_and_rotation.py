import json
import math
import numpy as np
import matplotlib.pyplot as plt

from Results_robo_scans.UKF_DENI_Class import UKFDeni
from Results_robo_scans.transofmation_vector_ukf import DeniTransformation


def coordinates_new_pos_car( x_base, y_base, angle, distance):
    """
    :param orientation_angle: which direction in the world is oriented the robot
    :param x_base: where is lidar on x
    :param y_base: where is lidar on y
    :param angle: in degree ,0 is +y
    :param distance: from lidar to the point
    :return: new coordiantes of the found point
    """

    new_x = distance * math.sin(math.radians(angle))
    new_y = distance * math.cos(math.radians(angle))
    final_x = x_base + new_x
    final_y = y_base + new_y
    final_x = round(final_x, 2)
    final_y = round(final_y, 2)
    if final_x == 0 or final_x == -0:
        final_x = 0
    if final_y == 0 or final_y == -0:
        final_y = 0
    return final_x, final_y


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
test_path=[(0,0),(0,10),(0,20),(0,70),(90.5,0)]
robot_coordinates=[(0,0),(0,10),(0,30),(0,100),(0,100)]

DT=DeniTransformation()
ukfdeni=UKFDeni()

with open("lidar_test_UKF_2.json",'r') as jso:
    land_all_points=json.load(jso)

list_with_coordiantes=[]
dict_node_coord={}
list_with_corrected_coordinates=[]
list_coord_without_ukf=[]

counter=0
center=0,0
end_x=0
end_y=0
a=0
for node,info in land_all_points.items():
    if counter>4:
        break
    cur_list = []
    new_places_=[]
    a = ukfdeni.localization(ukfdeni.start_x, ukfdeni.start_y, ukfdeni.start_theta, test_path[counter][0],
                    test_path[counter][1], info,sigma_vel=0.5,
            sigma_steer=np.radians(1), sigma_range=200, sigma_bearing=.01,step=1, ellipse_step=10)
    #print(a)
    for angle,dist in info:
        x,y=coordinates_for_x_y__from_distance_and_angle(
            a[2],a[1],a[0],angle,dist)
        x_, y_ = coordinates_for_x_y__from_distance_and_angle(
            test_path[counter][0], robot_coordinates[counter][0], robot_coordinates[counter][1], angle, dist)
        list_with_coordiantes.append((x,y))
        list_coord_without_ukf.append((x_,y_))
        cur_list.append((x,y))
    dict_node_coord[counter]=cur_list
    center_2 = center_of_gravity(cur_list)
    print(f'{counter} center bevor {center_2}')

    if counter==0:
        center = center_of_gravity(dict_node_coord[counter])
        new_places_=cur_list
    #print(f'first center {center}')
    if counter !=0:
        to_x=center_2[0]+center[0]
        to_y=center_2[1]+center[1]
        new_places_ = DT.translocation_new_version(cur_list,center_2[0],center_2[1],center[0],center[1], 0)
    center_ = center_of_gravity(new_places_)
    print(f'{counter} center after {center_}')
    list_with_corrected_coordinates.extend(new_places_)
    x=center_2[0]-center[0]
    y=center_2[1]-center[1]
    end_x+=x
    end_y+=y
    #print(f'{counter} step ===> x: {end_x}, y: {end_y}')


    counter+=1


landmarks=np.array(list_with_coordiantes)
landmarks_2=np.array(list_with_corrected_coordinates)
landmarks_3=np.array(list_coord_without_ukf)
rob=np.array(robot_coordinates)
##
landmarks_4= np.array(DT.translocation(list_coord_without_ukf,300,0, 0))






# show

plt.scatter(landmarks[:, 0], landmarks[:, 1], s=5,c='green',label=f' UKF with {len(landmarks)}')
plt.scatter(landmarks_2[:, 0], landmarks_2[:, 1], s=5,c='blue',label=f'with point matching')
#plt.scatter(landmarks_3[:, 0], landmarks_3[:, 1], s=5,c='red',label=f'pure comands')
plt.scatter(landmarks_4[:, 0], landmarks_4[:, 1], s=5,c='red',label=f'pure comands')
plt.scatter(rob[:, 0], rob[:, 1],c='black',label='coordinates robot') # all detected points

plt.plot(0,400,color='yellow',label=f"Orient: {ukfdeni.start_theta}")
plt.plot(0,400,color='yellow',label=f"X: {a[0]}")
plt.plot(0,400,color='yellow',label=f"Y: {a[1]}")
plt.title("Translocation tests")
plt.legend(loc='upper left')
plt.grid()
plt.show()
