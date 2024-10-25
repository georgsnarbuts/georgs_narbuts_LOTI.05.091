#! /usr/bin/env python3

from cmath import inf
from visualization_msgs.msg import Marker,MarkerArray
import roslib, sys, rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry,Path
from geometry_msgs.msg import Point,PoseStamped
import numpy as np
import math
import time
import cvxopt
from cvxopt import matrix, solvers
from scipy.interpolate import CubicSpline
#------------------------------------------

class potential_field:
    def __init__(self):
        rospy.init_node("potential_field")

        self.sample_rate = rospy.get_param("~sample_rate", 10)

        # Subscribe to the global planner using the move base package. The global plan is the path that the robot would ideally follow if 
        # there are no unknown/dynamic obstacles. In the videos this is highlighted by green color.
        self.global_path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.handle_global_path)

        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.handle_laser)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.handle_odom)
        
        # Subscribe to the goal topic to get the goal position given using rviz's 2D Navigation Goal option.
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.handle_goal)

        # Publish the potential field vector topic which will be subscribed by the command_velocity node in order to
        # compute velocities.
        self.potential_field_pub = rospy.Publisher("potential_field_vector", Point,queue_size=10)

        # We store the path data gotten from the global planner above and display it. We have written a custom publisher 
        # in order to get more flexibility while displaying the paths.
        self.global_path_pub = rospy.Publisher("global_path",Path,queue_size=10)

        # This is a publisher to publish the robot path. In the videos it is highlighted by red color.
        self.robot_path_pub = rospy.Publisher("robot_path",Path,queue_size=10)
        
        self.path_robot = Path()
        self.path_robot.header.frame_id = 'map'

        ## TODO Choose suitable values
        self.eta = 10 # scaling factor for repulsive force
        self.zeta = 5 # scaling factor for attractive force
        self.q_star = 5 # threshold distance for obstacles
        self.d_star = 2 # threshoild distance for goal


        self.tang_eta = 10
        self.t_star = 3

        self.xi = 1 

        self.laser = None
        self.odom = None
        self.goal = None

        self.path_data = Path()
        self.path_data.header.frame_id = 'map'
        
        self.position_x = []
        self.position_y = []
        self.position_all = []
        
        # Boolean variables used for proper display of robot path and global path
        self.bool_goal = False
        self.bool_path = False
#------------------------------------------

    def start(self):
        rate = rospy.Rate(self.sample_rate)
        while not rospy.is_shutdown():
            if(self.path_data):
                self.global_path_pub.publish(self.path_data)
           
            self.robot_path_publish()
            #net_force = self.compute_attractive_force() + self.compute_repulsive_force() + self.compute_tangent_field() ## What should be the net force?
            net_force = self.compute_cbf_force()
            #net_force = self.compute_pd()
            self.publish_sum(net_force[0],net_force[1])

            rate.sleep()

#------------------------------------------
    def robot_path_publish(self):
        if(self.odom):
            odom_data = self.odom
            if(self.bool_path == True):
                self.bool_path = False
                self.path_robot = Path()
                self.path_robot.header.frame_id = 'map'
            pose = PoseStamped()
            pose.header = odom_data.header
            pose.pose = odom_data.pose.pose
            self.path_robot.poses.append(pose)
            self.robot_path_pub.publish(self.path_robot)

#------------------------------------------

    def compute_repulsive_force(self):
        if (self.laser is None or self.odom is None or self.goal is None or self.path_data is None or not self.path_data.poses):
            return np.array([0, 0])

        laser_data = self.laser
        ranges = np.array(laser_data.ranges)
        angle = laser_data.angle_min  # Start from the min angle of the laser scan
        resolution = laser_data.angle_increment
        vector_sum = np.array([0.0, 0.0])

        for distance in ranges:
            if 0.1 < distance < self.q_star:  

                mag = self.eta * ((1 / distance) - (1 / self.q_star)) ** 2

                vector_x = mag * np.cos(angle)
                vector_y = mag * np.sin(angle)

                vector_sum[0] -= vector_x
                vector_sum[1] -= vector_y

            angle += resolution

        if len(ranges) > 0:
            vector_sum = vector_sum * (1 / len(ranges))

        return np.array([vector_sum[0], vector_sum[1]])
        



    def compute_tangent_field(self):
        if self.laser is None or self.odom is None or self.goal is None:
            return (0, 0)

        laser_data = self.laser
        ranges = np.array(laser_data.ranges)
        angle = laser_data.angle_min  
        resolution = laser_data.angle_increment
        vector_sum = np.array([0.0, 0.0])

        speed = np.hypot(self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y)
        start_point = (self.path_data.poses[0].pose.position.x, self.path_data.poses[0].pose.position.y)
        end_point = (self.path_data.poses[-1].pose.position.x, self.path_data.poses[-1].pose.position.y)

        pos_x = self.odom.pose.pose.position.x
        pos_y = self.odom.pose.pose.position.y
        current_point = (pos_x, pos_y)

        for distance in ranges:
            if 0.1 < distance < self.t_star:  
                mag = self.tang_eta * ((1 / distance) - (1 / self.t_star)) ** 2

                
                grad_x = mag * np.cos(angle)
                grad_y = mag * np.sin(angle)

                # turning the vector by 90 degrees
                tan_x = -grad_y
                tan_y = grad_x

                tan_mag = 1 * mag

                if speed < 1.5 and current_point != start_point and current_point != end_point:
                    vector_sum[0] += tan_mag * tan_x
                    vector_sum[1] += tan_mag * tan_y

            # Increment angle to the next reading
            angle += resolution

        # Normalize the vector sum by the number of laser readings
        if len(ranges) > 0:
            vector_sum = vector_sum * (1 / len(ranges))

        return np.array([vector_sum[0], vector_sum[1]])
    


#------------------------------------------
    
    def compute_attractive_force(self):
        if self.odom == None or self.goal == None:
            return (0,0)

        odom_data = self.odom
        pos_x = odom_data.pose.pose.position.x
        pos_y = odom_data.pose.pose.position.y
        pos = []
        pos.append(pos_x)
        pos.append(pos_y)

        closest_waypoint = []
        while(not closest_waypoint or closest_waypoint is None or not closest_waypoint[1]):
            closest_waypoint = self.closest_waypoint(pos, self.position_all)

        dist_to_goal = np.sqrt((pos_x - self.goal.pose.position.x)**2 + (pos_y - self.goal.pose.position.y)**2)

        index_waypoint = closest_waypoint[0]
        
        waypoint_x, waypoint_y = self.position_all[index_waypoint]
        
        if (dist_to_goal<=self.d_star):
            mag = self.zeta
        else:
            mag = (self.d_star * self.zeta)/dist_to_goal

        direction = [(self.goal.pose.position.x - pos_x), (self.goal.pose.position.y - pos_y)]
    
        vector = [mag * direction[0], mag * direction[1]]

        return np.array([vector[0],vector[1]])
    
    '''
    def path_spline(self, x_path, y_path):
        x_diff = np.diff(x_path)
        y_diff = np.diff(y_path)
        phi = np.unwrap(np.arctan2(y_diff, x_diff))
        phi_init = phi[0]
        phi = np.hstack((phi_init, phi))
        arc = np.cumsum(np.sqrt(x_diff**2+y_diff**2))
        arc_length = arc[-1]
        arc_vec = np.linspace(0, arc_length, np.shape(x_path)[0])
        cs_x_path = CubicSpline(arc_vec, x_path)
        cs_y_path = CubicSpline(arc_vec, y_path)
        cs_phi_path = CubicSpline(arc_vec, phi)
        cs_xdot_path = cs_x_path.derivative()
        cs_ydot_path = cs_y_path.derivative()
        return cs_x_path, cs_y_path, cs_phi_path, arc_length, arc_vec, cs_xdot_path, cs_ydot_path


    def waypoint_generator(self, x_global_init, y_global_init, x_path_data, y_path_data, arc_vec, cs_x_path, cs_y_path, cs_phi_path, arc_length, cs_xdot_path, cs_ydot_path):
        v = 1
        dt = 0.1

        idx = np.argmin(np.sqrt((x_global_init-x_path_data)
                        ** 2+(y_global_init-y_path_data)**2))
        arc_curr = arc_vec[idx]  # s_0
        arc_pred = arc_curr + v*dt  # s_0+sdot*delta_t from slides
        x_waypoints = cs_x_path(arc_pred)  # x_d(s_0+sdot*delta_t) from slides
        y_waypoints = cs_y_path(arc_pred)  # y_d(s_0+sdot*delta_t) from slides
        phi_Waypoints = cs_phi_path(arc_pred)

        xd_dot = cs_xdot_path(arc_pred)*v
        yd_dot = cs_ydot_path(arc_pred)*v

        # returns x_d, y_d, _ from slides
        return x_waypoints, y_waypoints, phi_Waypoints, xd_dot, yd_dot
    '''

    def path_spline(self, x_path, y_path):
        x_diff = np.diff(x_path)
        y_diff = np.diff(y_path)
        phi = np.unwrap(np.arctan2(y_diff, x_diff))
        phi_init = phi[0]
        phi = np.hstack(( phi_init, phi  ))
        arc = np.cumsum( np.sqrt( x_diff**2+y_diff**2 )   )
        arc_length = arc[-1]
        arc_vec = np.linspace(0, arc_length, np.shape(x_path)[0])
        cs_x_path = CubicSpline(arc_vec, x_path)
        cs_y_path = CubicSpline(arc_vec, y_path)
        cs_phi_path = CubicSpline(arc_vec, phi)
        cs_xdot_path = cs_x_path.derivative()
        cs_ydot_path = cs_y_path.derivative()
        return cs_x_path, cs_y_path, cs_phi_path, arc_length, arc_vec, cs_xdot_path, cs_ydot_path



    def waypoint_generator(self, x_global_init, y_global_init, x_path_data, y_path_data, arc_vec, cs_x_path, cs_y_path, cs_phi_path, arc_length, cs_xdot_path, cs_ydot_path):
        v = 1
        dt = 0.1
        idx = np.argmin( np.sqrt((x_global_init-x_path_data)**2+(y_global_init-y_path_data)**2))
        arc_curr = arc_vec[idx] ## s_0
        arc_pred = arc_curr + v*dt # s_0+sdot*delta_t from slides
        x_waypoints = cs_x_path(arc_pred) ### x_d(s_0+sdot*delta_t) from slides
        y_waypoints =  cs_y_path(arc_pred)### y_d(s_0+sdot*delta_t) from slides
        phi_Waypoints = cs_phi_path(arc_pred)
        
        xd_dot = cs_xdot_path(arc_pred)
        yd_dot = cs_ydot_path(arc_pred)
        
        return x_waypoints, y_waypoints, phi_Waypoints, xd_dot, yd_dot ## returns x_d, y_d, _ from slides

    def compute_pd(self):

        #if self.odom == None or self.goal == None:
        #    return (0,0)

        k_p = 0.6
        k_d = 0.2


        

        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        x_dot = self.odom.twist.twist.linear.x
        y_dot = self.odom.twist.twist.linear.y

        loaded_array = np.load(
            "/home/georgs/Downloads/circle.npy")

        # Split the two arrays into x and y paths
        x_path = loaded_array[0]  
        y_path = loaded_array[1] 



        if self.goal:

            cs_x_path, cs_y_path, cs_phi_path, arc_length, arc_vec, cs_xdot_path, cs_ydot_path = self.path_spline(x_path, y_path)

            x_waypoints, y_waypoints, phi_Waypoints, xd_dot, yd_dot = self.waypoint_generator(x, y, x_path, y_path, arc_vec, cs_x_path, cs_y_path, cs_phi_path, arc_length, cs_xdot_path, cs_ydot_path)

            vel_x = k_d * (xd_dot - x_dot) + k_p * (x_waypoints - x)

            vel_y = k_d * (yd_dot - y_dot) + k_p * (y_waypoints - y) 

            return np.array([vel_x, vel_y])

            

#------------------------------------------
# cylinder at ccoords 0.910302 -4.25107
    def compute_cbf_force(self):
        if (self.laser is None or self.odom is None or self.goal is None or
            self.path_data is None or not self.path_data.poses):
            return np.array([0, 0])

        # laser data
        laser_data = self.laser
        ranges = np.array(laser_data.ranges)
        angle = laser_data.angle_min  # Start from the min angle of the laser scan
        resolution = laser_data.angle_increment

        # odometry data
        odom_data = self.odom
        x = odom_data.pose.pose.position.x
        y = odom_data.pose.pose.position.y

        # Goal position
        x_goal = self.goal.pose.position.x
        y_goal = self.goal.pose.position.y

        alpha = 0.6

        k_att = 1

        num_dim = 2
        
        #v_des = np.hstack(( -k_att*(x-x_goal), -k_att*(y-y_goal)))
        v_des = self.compute_pd()
        Q = np.identity(num_dim)
        q = -v_des
        A_in = []
        b_in = []
        
        for distance in ranges:
            if np.isfinite(distance):
                A = [np.cos(angle), np.sin(angle)]
                B = [alpha * (distance-0.2)] 

                A_in.append(A)
                b_in.append(B)


            angle += resolution


        A_in = np.vstack(A_in)
        b_in = np.vstack(b_in)
        
            
        sol_data = solvers.qp( cvxopt.matrix(Q, tc = 'd'), cvxopt.matrix(q, tc = 'd'), cvxopt.matrix(A_in, tc = 'd'), cvxopt.matrix(b_in, tc = 'd'), None, None  )

        sol = np.asarray(sol_data['x'])

        return sol

    

#------------------------------------------

    def closest_waypoint(self,point, points):
        i=0
        pt=[]
        dist = math.inf
        for p in points:
            if(math.dist(p,point)<dist):
                dist = math.dist(p,point)
                pt = p
                i = points.index(pt)
        return [i,pt]
#------------------------------------------
    def handle_laser(self, laser_data):
        self.laser = laser_data
        
#------------------------------------------
    def handle_odom(self, odom_data):
        self.odom = odom_data
#------------------------------------------ 
    def handle_goal(self, goal_data):
        self.bool_goal = True
        self.bool_path = True
        self.goal = goal_data
#------------------------------------------
    def publish_sum(self, x, y):
        vector = Point(x, y, 0)
        self.potential_field_pub.publish(vector)
#------------------------------------------
    def publish_dist_to_goal(self, dist):
        dist_to_goal = Float32(dist)
        self.dist_to_goal_pub.publish(dist_to_goal)
#------------------------------------------

    def handle_global_path(self, path_data):
        if(self.bool_goal == True):
            self.bool_goal = False
            self.path_data = path_data
            i=0
            while(i <= len(self.path_data.poses)-1):
                self.position_x.append(self.path_data.poses[i].pose.position.x)
                self.position_y.append(self.path_data.poses[i].pose.position.y)
                i=i+1
            self.position_all = [list(double) for double in zip(self.position_x,self.position_y)]
            self.position_x = []
            self.position_y = []

if __name__ == "__main__":
    pf = potential_field()
    pf.start()
