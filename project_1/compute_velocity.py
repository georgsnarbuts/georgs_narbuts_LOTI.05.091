#! /usr/bin/env python3

import roslib, sys, rospy
from sensor_msgs.msg import LaserScan, Joy
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Float32
import numpy as np
import math
from nav_msgs.msg import Odometry

#-------------------------------------------------

class safe_teleop:
    def __init__(self):
        rospy.init_node("command_velocity")

        rospy.Subscriber("potential_field_vector", Point, self.handle_potential_field)
        self.cmd_pub = rospy.Publisher("cmd_vel_robotont", Twist,queue_size=10)

        self.odom = None
        self.obstacle_vector = None
        self.min_vel_x = -1
        self.max_vec_x = 1
        self.min_vel_y = -1
        self.max_vec_y = 1
        self.drive_scale = 5 # scaling factor to scale the net force

#-------------------------------------------------

    def start(self):
        rate = rospy.Rate(rospy.get_param("~cmd_rate", 10))
        while not rospy.is_shutdown():
            cmd = self.compute_motion_cmd()
            if cmd != None:
                self.cmd_pub.publish(cmd)
            rate.sleep()

#-------------------------------------------------

    def compute_motion_cmd(self):
        if (self.obstacle_vector is None):
            cmd = None
        else:
            cmd = Twist()

            # We use velocity based potential field,that is, the gradient/force is directly commanded as velocities
            # instead of force or acceleration. 

            vel_x = self.obstacle_vector[0]*self.drive_scale
            vel_y = self.obstacle_vector[1]*self.drive_scale

            velocity_magnitude = np.sqrt(vel_x**2 + vel_y**2)
            vel_clipped = np.clip(velocity_magnitude, 0, 1)
            # cmd.linear.x = np.clip(vel_x,self.min_vel_x,self.max_vec_x)
            # cmd.linear.y = np.clip(vel_y,self.min_vel_y,self.max_vec_y)

            theta = np.arctan2(vel_y, vel_x)

            velx = vel_clipped * np.cos(theta)
            vely = vel_clipped * np.sin(theta)

            # clipped_velocity_mag_x = np.clip(velocity_magnitude, self.min_vel_x,self.max_vec_x)
            # clipped_velocity_mag_y = np.clip(velocity_magnitude, self.min_vel_y,self.max_vec_y)

            cmd.linear.x = velx
            cmd.linear.y = vely


            
        return cmd

#-------------------------------------------------

    def handle_potential_field(self, potential_field):
        self.obstacle_vector = np.array([potential_field.x, potential_field.y])

if __name__ == "__main__":
    st = safe_teleop()
    st.start()