import math

import numpy as np
from matplotlib import pyplot as plt


class DeniTransformation:
    def __init__(self):
        pass
    def calc_sum_of_vectors(self,v_1,v_2):
        """ sum of two vectors"""
        result=v_1+v_2
        print(result)
        x,y=result[0],result[1]
        return result

    def scaling_vector(self,vector,scalers):
        """ multiply vector by some value"""
        new_vecotor=vector*scalers
        return new_vecotor

    def rotation_base_vectors_by_angle(self,theta):
        """
        :param theta: angle in degrees
        :return: new coord of i,j bases vectors
        """
        theta=np.radians(theta)
        new_i=[np.cos(theta),np.sin(theta)]
        new_j=[-np.sin(theta),np.cos(theta)]
        return new_i,new_j

    def translocation(self,coordinates,new_x,new_y,teta):
        """
        :param teta: angle which to turn
        :param coordinates: cur_coor of all points
        :param new_x: where in x
        :param new_y: where in y
        :return:
        """
        end = np.insert(coordinates, 2, 1, axis=1) # add a Z axis
        teta=np.radians(teta)
        trans_matrix = np.array([
            [np.cos(teta),np.sin(teta), 0],
            [-np.sin(teta),np.cos(teta), 0],
            [-new_x, -new_y, 1]])
        result_homog_matrix=np.dot(end,trans_matrix)
        result_end=np.delete(result_homog_matrix,2,axis=1) # remove Z axis
        return result_end


    def translocation_new_version(self,coordinates,new_x,new_y,teta):
        """
        :param teta: angle which to turn
        :param coordinates: cur_coor of all points
        :param new_x: where in x
        :param new_y: where in y
        :return:
        """
        # x=new_x-(new_x-from_x)
        # y=new_y-(new_y-from_y)
        x=new_x
        y=new_y
        end = np.insert(coordinates, 2, 1, axis=1) # add a Z axis
        teta=np.radians(teta)
        trans_matrix = np.array([
            [np.cos(teta),-np.sin(teta), x],
            [np.sin(teta),np.cos(teta), y],
            [0, 0, 1]])
        result_homog_matrix=np.dot(end,trans_matrix)
        result_end=np.delete(result_homog_matrix,2,axis=1) # remove Z axis
        return result_end

    def linear_transformation(self,coordinates_point:list, angle):
        """ [x,y] input
        return the coord of the vector(point), by transformation of the coord frame acording the angle """
        new_coord_of_base_i=[1,0] # show how is the coordinate frame is rotated
        new_coord_of_base_j=[1,1.5] # show how is the coordinate frame is rotatedrotated
        new_coord_of_base_i,new_coord_of_base_j=self.rotation_base_vectors_by_angle(angle)
        # new_coord_of_base_i=translocation(new_coord_of_base_i,1,1)
        # new_coord_of_base_j=translocation(new_coord_of_base_j,1,1)
        new_coordinates=(np.array(new_coord_of_base_i))*coordinates_point[0] + (np.array(new_coord_of_base_j))*coordinates_point[1]
        #new_coordinates=np.array(scaling_vector(new_coord_of_base_i,2))*coordinates_point[0]  + np.array(scaling_vector(new_coord_of_base_j,2))*coordinates_point[1]
        return new_coordinates

smile=[[0.5,2],[-0.5,2],[0,0],[-0.5,0],[0.5,0],[-1,0],[1,0],[-2,0.5],[2,0.5],[-1.5,0.25],[1.5,0.25]]
ang=90.9
to_x=-10
to_y=0
color='green'
point_size=30
dimensions_world_frame=6
world_frame = np.array([
    [0, 0],
    [0, dimensions_world_frame],
    [dimensions_world_frame, 0],
    [dimensions_world_frame,dimensions_world_frame],
    [dimensions_world_frame,-dimensions_world_frame],
    [-dimensions_world_frame,-dimensions_world_frame],
    [-dimensions_world_frame,dimensions_world_frame],
    [-dimensions_world_frame,0],
    [0,-dimensions_world_frame],

])
data=np.array(smile)
DT=DeniTransformation()
new_places_=DT.translocation(data,to_x,to_y,ang)
data_transf=np.array(new_places_)
x,y=data.T
x_t,y_t=data_transf.T
x_w,y_w=world_frame.T
# fig, ax = plt.subplots()
# ax.scatter(x, y, color=f'{color}', s=point_size)
# ax.scatter(x_t, y_t, color=f'red', s=point_size)
# ax.scatter(0, 0, color='black', s=point_size)
# ax.scatter(x_w, y_w, color='black', s=point_size)
# #ax.annotate('transpose', (x_t, y_t),fontsize=10,color='red')
# plt.grid()
# fig.show()
