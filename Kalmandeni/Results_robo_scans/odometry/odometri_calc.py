import math

od_l=100
od_r=100

def tttt(l,r,teta):
    rad=3.64
    L=22
    x=(rad/2)*(l+r)*math.cos(teta)
    y=(rad/2)*(l+r)*math.sin(teta)
    theta=(rad/L)*(l+r)
    return (x,y,math.degrees(theta))



def calc_orient(current_theta_angle,width_of_car,vel_l, vel_r):
    """
    this can work only for going ahead
    this function calculate the actual angle of orientation of the car in the world
    :return: current actual theta angle in degrees
    """
    for steps in range(1):
        delta_time =1
        current_theta_angle += (vel_l + vel_r) / width_of_car * delta_time
        if current_theta_angle >= 2 * math.pi:
            current_theta_angle = 0
    return math.degrees(current_theta_angle)


#res=calc_orient(0,25,1,1)
#print(res)


res=tttt(1,4,0)
print(res)