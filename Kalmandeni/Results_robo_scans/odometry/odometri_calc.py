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



def real_yaw_angle_in_degree(vel,divider_factor,sum_angular_velocity):
    """

    :param vel: current velocity
    :param divider_factor: how to reduce the error in measuring 0.001
    :param sum_angular_velocity: all summed velocity
    :return: return the angle of turning in degrees
    """
    current_angular_velocity = vel
    sum_angular_velocity += abs(current_angular_velocity)
    real_yaw_angle_degrees = sum_angular_velocity // divider_factor
    return real_yaw_angle_degrees




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



def odometry_4_wheels(vel_l,vel_r,vel_l_2,vel_r_2,orientation):
    # calc theta
    R_w=0.125 # widht of base in m
    r=3.4 # radus wheel in cm
    new_theta=(r/2*R_w)*(vel_r-vel_l)
    new_theta_2=(r/2*R_w)*(vel_r_2-vel_l_2)
    sum_theta=(new_theta+new_theta_2)/2

    # calc the position
    v_1=(vel_l+vel_r)/2
    v_2=(vel_l_2+vel_r_2)/2
    v=(v_1+v_2)/2
    x_=v*math.cos(math.radians(orientation))
    y_=v*math.sin(math.radians(orientation))
    return sum_theta,x_,y_


def odometry(vel_r,vel_l,orientation):
    # calc theta
    R_w=0.125 # widht of base
    r=3.4 # radus wheel
    new_theta=(r/2*R_w)*(vel_r-vel_l)

    # calc the position
    v=(vel_l+vel_r)/2
    x_=v*math.cos(math.radians(orientation))
    y_=v*math.sin(math.radians(orientation))
    return new_theta,x_,y_



def odometry_distance(dist_r,dist_l,orientation):
    vel_r,vel_l=1,1
    # calc theta
    R_w=0.25 # widht of base
    r=3.39 # radus wheel
    new_theta=(r/2*R_w)*((vel_r*dist_r)-(vel_l*dist_l))
    new_theta = abs(math.degrees(new_theta))
    if new_theta>360:
        new_theta-=360

    # calc the position
    v=(vel_l+vel_r)/2
    x_=v*math.cos(math.radians(orientation))
    y_=v*math.sin(math.radians(orientation))
    return new_theta,x_,y_

start_theta=0
start_x=0
start_y=0
start_theta_2=0
start_x_2=0
start_y_2=0
current_theta=0

speed_test=[[1.4,-1,1.5,-0.9],[-1,1,-1,0.9],[1,1,1,1],[1.4,1,-1.5,-0.9],[-1,1,-1,0.9],[-1,1,-1,1],[-1.4,1,-1.5,0.9],[-1,1,-1,0.9],[-1,1,-1,1],[1,-1,1,-1]]
for step in range(len(speed_test)):
    """ A---B
          |
        C---D  
    """
    speed_a=speed_test[step][0]
    speed_b=speed_test[step][1]
    speed_c=speed_test[step][2]
    speed_d=speed_test[step][3]

    theta,x,y=odometry(speed_a,speed_b,start_theta)
    theta_2,x_2,y_2=odometry_4_wheels(speed_a,speed_b,speed_c,speed_d,start_theta_2)
    start_theta+=theta
    start_x+=x
    start_y+=y
    start_theta_2 += theta_2
    start_x_2 += x_2
    start_y_2 += y_2
    if math.degrees(start_theta)>360 or math.degrees(start_theta)<-360:
        start_theta=0
   # print(f'2 wheels theta= {abs(math.degrees(start_theta)):.3f}, x= {start_x:.3f} , y= {start_y:.3f}')
    print(f'4 WHEELS theta= {abs(math.degrees(start_theta_2)):.3f}, x= {start_x_2:.3f} , y= {start_y_2:.3f}')


# theta,x,y=odometry_distance(10,-10,start_theta)
# print(f'deg= {theta:.3f}, x= {x:.3f} , y= {y:.3f}')





