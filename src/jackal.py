import math

import torch


class Jackal:
    def __init__(self):
        self.dtype = torch.float

        self.linear_velocity = 1.5
        self.x = 0.0
        self.y = 0.0
        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0
        self.q4 = 0.0
        self.current_angle = 0.0
        self.control_law = 0.0

        self.batch_size = 128

        self.robot_path = []

    def get_pose(self):
        return self.x, self.y

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
