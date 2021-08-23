import numpy as np


class Path:
    def __init__(self, points):
        self.path = points
        self.stop = False
        self.path_length = len(self.path) - 1
        self.path_count = 0

        self.estimated_path_length = self.calcualte_estimated_path_length()

    def calcualte_estimated_path_length(self):
        length = 0

        prev = self.path[0]

        for i in range(1, len(self.path)):
            curr = self.path[i]

            prev = curr

    def get_trajectory(self, robot):
        pos_x, pos_y = robot.get_pose()

        current_point = np.array(self.path[self.path_count])
        target = np.array(self.path[self.path_count + 1])

        A = np.array([[(current_point[1] - target[1]), (target[0] - current_point[0])],
                      [(target[0] - current_point[0]), (target[1] - current_point[1])]])
        b = np.array([[(target[0] * current_point[1] - current_point[0] * target[1])],
                      [(pos_x * (target[0] - current_point[0]) + pos_y * (target[1] - current_point[1]))]])
        proj = np.matmul(np.linalg.inv(A), b)

        current_point = np.array([[current_point[0]], [current_point[1]]])
        target = np.array([[target[0]], [target[1]]])
        temp1 = proj - current_point  ####dot product
        temp2 = target - current_point
        proj_len = (temp1[0] * temp2[0] + temp1[1] * temp2[1]) / np.linalg.norm(target - current_point, 2) ** 2

        if proj_len > 1:
            self.path_count += 1

        if self.path_count == self.path_length - 1:
            self.stop = True

        if (self.path_count == (self.path_length - 2)) or (self.path_count == (self.path_length - 1)):
            curr = np.array(self.path[self.path_count])
            tar = np.array(self.path[self.path_count + 1])
            future = np.array(self.path[self.path_count + 1])
        else:
            curr = np.array(self.path[self.path_count])
            tar = np.array(self.path[self.path_count + 1])
            future = np.array(self.path[self.path_count + 2])

        robot.stop = self.stop

        return curr, tar, future, self.stop
