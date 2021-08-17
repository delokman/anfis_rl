import math

import rospy
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class Jackal:
    def __init__(self):
        self.dtype = torch.float

        self._linear_velocity = 1.5
        self.x = 0.0
        self.y = 0.0
        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0
        self.q4 = 0.0
        self.current_angle = 0.0
        self._control_law = 0.0

        self.batch_size = 128

        self.robot_path = []

    def get_pose(self):
        return self.x, self.y

    @property
    def linear_velocity(self):
        return self._linear_velocity

    @linear_velocity.setter
    def linear_velocity(self, new_v):
        self._linear_velocity = new_v
        self.twist.linear.x = self._linear_velocity

    @property
    def control_law(self):
        return self._linear_velocity

    @control_law.setter
    def control_law(self, new_v):
        self._control_law = new_v
        self.twist.angular.z = self._control_law

    def callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.q2 = msg.pose.pose.orientation.x
        self.q3 = msg.pose.pose.orientation.y
        self.q4 = msg.pose.pose.orientation.z
        self.q1 = msg.pose.pose.orientation.w
        self.current_angle = math.atan2(2 * (self.q1 * self.q4 + self.q2 * self.q3),
                                        1 - 2 * (self.q3 ** 2 + self.q4 ** 2))

        if not self.stop:
            self.robot_path.append([self.x, self.y])
        print("X:", self.x, " Y:", self.y, " Angle:", math.degrees(self.current_angle))
