import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def z_course(segment_length, start_angle=0, end_angle=180, step_angle=15):
    num = int(np.ceil((end_angle - start_angle) / step_angle)) + 1
    print(num)

    step_start = int(np.floor(start_angle / step_angle))
    start_angle = np.radians(start_angle)
    step_angle = np.radians(step_angle)

    points = [[0., 0.], [segment_length, 0]]
    angles = [0, start_angle]

    direction = -1

    print(step_start)

    for i in range(1, num):
        xp, yp = points[i]
        tp = angles[i]

        x = xp + segment_length * np.cos(tp)
        y = yp + segment_length * np.sin(tp)
        t = angles[i] + direction * (i + step_start) * step_angle

        direction *= -1

        points.append([x, y])
        angles.append(t)

    print(np.degrees(angles))

    return points


if __name__ == '__main__':
    points = z_course(12, 45, 180)

    points = np.array(points)

    plt.plot(points[:, 0], points[:, 1])
    for i in range(points.shape[0] - 1, 0, -1):
        plt.scatter(points[i, 0], points[i, 1], s=(i + 1) * 10)
    a: Axes = plt.gca()
    a.set_aspect('equal')
    plt.show()
