import numpy as np


def extend_path(path: list):
    """
    In order to fix angle errors and index out of bounds, the path is extended by a single segment which is in the same angle as the last segment

    Args:
        path: The path to extend
    """
    before_end, end = np.array(path[-2]), np.array(path[-1])

    diff = (end - before_end)
    diff /= np.linalg.norm(diff)

    after_end = diff * 10 + end

    path.append(after_end)


class Path:
    def __init__(self, points):
        self.path = points
        self.stop = False
        self.path_length = len(self.path) - 1
        self.path_count = 0

        self.estimated_path_length = self.calcualte_estimated_path_length()

        self.transform = None

    def reset(self):
        self.stop = False
        self.path_count = 0
        self.transform = None

    def calcualte_estimated_path_length(self):
        length = 0

        prev = self.path[0]

        for i in range(1, len(self.path) - 1):
            curr = self.path[i]

            length += np.linalg.norm(np.subtract(prev, curr))

            prev = curr

        return length

    def get_estimated_time(self, linear_velocity):
        return self.estimated_path_length / linear_velocity

    def get_trajectory(self, robot, progress_bar=None):
        pos_x, pos_y = robot.get_pose()

        continue_search = True

        while continue_search:

            current_point = np.array(self.path[self.path_count])
            target = np.array(self.path[self.path_count + 1])

            if self.transform is not None:
                current_point = self.transform @ np.append(current_point, 1)
                target = self.transform @ np.append(target, 1)

                current_point = current_point[:-1]
                target = target[:-1]

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

                if progress_bar is not None:
                    progress_bar.update()
            else:
                continue_search = False

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

        if self.transform is not None:
            curr = self.transform @ np.append(curr, 1)
            tar = self.transform @ np.append(tar, 1)
            future = self.transform @ np.append(future, 1)

            curr = curr[:-1]
            tar = tar[:-1]
            future = future[:-1]

        robot.stop = self.stop

        return curr, tar, future, self.stop

    def transformed_path(self):
        if self.transform is None:
            return self.path
        else:
            points = []

            for p in self.path:
                p = self.transform @ np.append(p, 1)
                p = p[:-1]
                points.append(p)

            return points

    def set_initial_state(self, jackal, y_shift=0):
        pose = jackal.get_pose()
        theta = jackal.get_angle()
        print(pose, theta)
        #
        # T = np.array([
        #     [1, 0, pose[0]],
        #     [0, 1, pose[1]],
        #     [0, 0, 1],
        # ])
        #
        # R = np.array([
        #     [np.cos(theta), -np.sin(theta), 0],
        #     [np.sin(theta), np.cos(theta), 0],
        #     [0, 0, 1],
        # ])

        T = np.array([
            [1, 0, pose[0]],
            [0, 1, pose[1] + y_shift],
            [0, 0, 1],
        ])

        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])

        self.transform = T @ R

        T = np.array([
            [1, 0, -pose[0]],
            [0, 1, -pose[1] - y_shift],
            [0, 0, 1],
        ])

        R = np.array([
            [np.cos(-theta), -np.sin(-theta), 0],
            [np.sin(-theta), np.cos(-theta), 0],
            [0, 0, 1],
        ])

        self.inverse_transform = R @ T

    def inverse_transform_poses(self, path):
        path = np.asarray(path)
        path = np.append(path, np.zeros((path.shape[0], 1)), axis=1)
        poses = (path @ self.inverse_transform[:, :, None]).squeeze(-1).T

        return poses
