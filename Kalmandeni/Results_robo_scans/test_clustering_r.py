import math
import random

import numpy as np
from matplotlib import pyplot as plt_source
import time

class VisualClustering:
    def __init__(self):
        self.plt = plt_source
        self.font_size_node_text = 20
        self.size_node_point = 20
        self.point_size = 10
        self.color_node = 'black'
        self.test_points={}
        self.test_points_old={}
        self.line=[(1,1),(5,5)]
        self.k_1=(1,1) # k factor 1
        self.k_2=(5,5) # k factor 2
    def visual_this_points(self, cluster_line, color='green'):
        """
        visualisation the all info in the cluster class
        :param points:
        :param cluster_line:
        :param color:
        :return:
        """

        poin_1=[]
        poin_2=[]
        cluster_centers=[(self.k_1[0],self.k_1[1]),(self.k_2[0],self.k_2[1])]
        for point, info in self.test_points.items():
            if info['color']=='red':
                poin_1.append((info['coord'][0],info['coord'][1]))
            else:
                poin_2.append((info['coord'][0], info['coord'][1]))



        data_group_1=np.array(poin_1)
        data_group_2=np.array(poin_2)
        cluster=np.array(cluster_centers)
        all_x=[]
        all_y=[]
        for x,y in cluster_line:
            all_x.append(x)
            all_y.append(y)

        x,y=data_group_1.T
        no=False
        try:
            x_,y_=data_group_2.T
        except:
            no=True

        x_cl,y_cl=cluster.T
        fig, ax = self.plt.subplots()
        ax.scatter(x, y, color=f'{"red"}', s=self.point_size)
        if not no:
            ax.scatter(x_, y_, color="blue", s=self.point_size)
        ax.scatter(x_cl, y_cl, color=f'{self.color_node}', s=self.size_node_point)
        self.plt.plot(all_x, all_y, 'k')
        fig.show()
        return

    def generate_random_points(self):
        """
        generate random dikt with points and same color
        :return:
        """
        count_points = 10000
        return_for=[]
        for point in range(count_points):
            #self.test_points.append((random.randint(1, 20), random.randint(1, 20)))
            a,b=random.randint(1, 50), random.randint(1, 50)
            self.test_points[point]={}
            self.test_points[point]['coord']=((a,b))
            self.test_points[point]['color']='white'
            return_for.append((a,b))

        return return_for

    def check_if_group_points_are_same(self):
        """
        check if current points are same as points last calculation
        :return:
        """
        for each_point,info in self.test_points.items():
            try:
                if info['coord'] != self.test_points_old[each_point]['coord']:
                    return False
                if info['color'] != self.test_points_old[each_point]['color']:
                    return False
            except:
                return False
        return True


    def calc_distance_between_two_points(self,a_x,a_y,b_x,b_y):
        """ this func calcualte the distance between two point in coord frame"""
        dist= math.sqrt(((max(b_x,a_x)-min(b_x,a_x))*2)+((max(b_y,a_y)-min(b_y,a_y))*2))
        return dist


    def calculate_geometric_center_group_of_points(self,points:list):
        """
        :param points: list of points in one group
        :return: new x,y coordinates of the geometric center of the points
        """
        x,y=0,0
        try:
            x = (1/len(points))*np.sum(points,axis=0)[0]
            y = (1 / len(points)) * np.sum(points, axis=0)[1]
        except:
            pass
        return x,y


    def calculation_k_means_clusterisation(self):
        """ main K means cluster function"""
        while True:
            if self.check_if_group_points_are_same():
                break
            for each_point,info in self.test_points.items():
                if self.calc_distance_between_two_points(self.k_1[0],self.k_1[1],info['coord'][0],info['coord'][1])>\
                    self.calc_distance_between_two_points(self.k_2[0], self.k_2[1], info['coord'][0], info['coord'][1]):
                    self.test_points[each_point]['color']='red'
                else:
                    self.test_points[each_point]['color'] = "blue"
            self.test_points_old=self.test_points.copy()
            self.k_1=self.calculate_geometric_center_group_of_points(
                [x['coord'] for x in self.test_points.values() if x['color']=='red'])
            self.k_2 = self.calculate_geometric_center_group_of_points(
                [x['coord'] for x in self.test_points.values() if x['color'] == 'blue'])
        return

    def calc_perpendicular_of_k_centers(self):
        mid_point=self.calc_distance_between_two_points(self.k_1[0],self.k_1[1],self.k_2[0],self.k_2[1])/2
        print(mid_point)
        hypo=7
        angle_hypotenusa=math.degrees(math.acos(mid_point/hypo))
        lenght_mid_line=math.sin(math.radians(angle_hypotenusa))*hypo
        print(angle_hypotenusa)
        print(lenght_mid_line)
        x_0=self.k_1[0]+(hypo/(mid_point*2))*(self.k_2[0]-self.k_1[0])*math.cos(math.radians(angle_hypotenusa))-(hypo/(mid_point*2))*(self.k_2[1]-self.k_1[1])*math.sin(math.radians(angle_hypotenusa))
        y_0=self.k_1[1]+(hypo/(mid_point*2))*(self.k_2[1]-self.k_1[1])*math.cos(math.radians(angle_hypotenusa)) + (hypo/(mid_point*2))*(self.k_2[0]-self.k_1[0])*math.sin(math.radians(angle_hypotenusa))
        print(x_0,y_0)
        x_1 = self.k_1[0] + (hypo / (mid_point * 2)) * (self.k_2[0] - self.k_1[0]) * math.cos(
            math.radians(angle_hypotenusa)) + (hypo / (mid_point * 2)) * (self.k_2[1] - self.k_1[1]) * math.sin(
            math.radians(angle_hypotenusa))
        y_1 = self.k_1[1] + (hypo / (mid_point * 2)) * (self.k_2[1] - self.k_1[1]) * math.cos(
            math.radians(angle_hypotenusa)) - (hypo / (mid_point * 2)) * (self.k_2[0] - self.k_1[0]) * math.sin(
            math.radians(angle_hypotenusa))
        print(x_1, y_1)
        self.line[0]=(x_0,y_0)
        self.line[1]=(x_1,y_1)
        return self.line







vis=VisualClustering()
vis.generate_random_points()
vis.calculation_k_means_clusterisation()
vis.visual_this_points(vis.calc_perpendicular_of_k_centers())

# #print(vis.calc_distance_between_two_points(1,1,3,3))
# vis.generate_random_points()
# vis.calculation_k_means_clusterisation()
# for each,info in vis.test_points.items():
#     print(info)

# t=vis.calculate_geometric_center_group_of_points([(1,1),(2,2)])
# print(t)

# a=np.sum([(1,1),(1,4)],axis=0)[1]
# print(a)