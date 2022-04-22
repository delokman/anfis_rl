import bezier
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


def straight_line(d=10, n=2):
    p = []
    for i in np.linspace(0, d, num=n):
        p.append([i, 0])
    return p


def curved_z(l, r, n=5):
    points = [
        [0, 0],
        [l, 0]
    ]

    cx, cy = points[-1]
    cy -= r

    dx = np.pi / n

    for i in np.linspace(np.pi / 2 - dx, -np.pi / 2, num=n):
        points.append([cx + r * np.cos(i), cy + r * np.sin(i)])

    points.append([0, -r * 2])

    cx, cy = points[-1]
    cy -= r

    for i in np.linspace(np.pi / 2 + dx, 3 * np.pi / 2, num=n):
        points.append([cx + r * np.cos(i), cy + r * np.sin(i)])

    points.append([l, -4 * r])

    return points


def courses_double_circle(n, r1=0.5, r2=1.):
    points = [
        [0, 0],
    ]

    shift = 2

    cx, cy = shift, r1

    dx = np.pi / n

    for i in np.linspace(-np.pi / 2 + dx, np.pi / 2, num=n):
        points.append([cx + r1 * np.cos(i), cy + r1 * np.sin(i)])

    cx, cy = shift, r1 * 3
    for i in np.linspace(3 * np.pi / 2 - dx, np.pi / 2, num=n):
        points.append([cx + r1 * np.cos(i), cy + r1 * np.sin(i)])

    cx, cy = shift, r1 * 4 - r2
    for i in np.linspace(np.pi / 2 - dx, -np.pi / 2, num=n):
        points.append([cx + r2 * np.cos(i), cy + r2 * np.sin(i)])

    cx, cy = shift, r1 * 4 - r2 * 3
    for i in np.linspace(-3 * np.pi / 2 + dx, -np.pi / 2, num=n):
        points.append([cx + r2 * np.cos(i), cy + r2 * np.sin(i)])

    return points


def random_points(n_point=100, n_segments=100):
    p = [
        np.array([0, 0]),
        np.array([1, 0])
    ]

    for i in range(n_point):
        pprev = p[-2]
        prev = p[-1]

        t_prev = np.arctan2(prev[1] - pprev[1], prev[0] - pprev[0])

        t = np.random.normal(0, np.deg2rad(90)) + t_prev
        r = np.random.normal(1, .5)

        x = np.cos(t) * r
        y = np.sin(t) * r

        p.append(prev + np.array([x, y]))

    points = np.array(p)
    curve = bezier.curve.Curve.from_nodes(points.T)

    return curve.evaluate_multi(np.linspace(0, 1, n_segments)).T


if __name__ == '__main__':
    points = z_course(12, 45, 180)

    points = np.array(points)

    plt.plot(points[:, 0], points[:, 1])
    for i in range(points.shape[0] - 1, 0, -1):
        plt.scatter(points[i, 0], points[i, 1], s=(i + 1) * 10)
    a: Axes = plt.gca()
    a.set_aspect('equal')
    plt.show()
