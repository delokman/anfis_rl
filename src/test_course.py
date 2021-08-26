import numpy as np


def test_course():
    path = [[0.0, 0.0], [1.0000, 0], [1.7000, -0.1876], [2.2124, -0.7000],
            [2.4000, -1.4000], [2.5876, -2.1000], [3.1000, -2.6124],
            [3.8000, -2.8000], [4.5000, -2.6124], [5.0124, -2.1000],
            [5.2000, -1.4000], [5.3876, -0.7000], [5.9000,
                                                   -0.1876], [6.6000, 0],
            [7.3000, -0.1876], [7.8124, -0.7000], [8.0000, -1.4000],
            [8.1876, -2.1000], [8.7000, -2.6124], [9.4000, -2.8000],
            [10.1000, -2.6124], [10.6124, -2.1000], [10.8000, -1.4000],
            [10.9876, -0.7000], [11.5000, -0.1876], [12.2000, 0],
            [12.9000, -0.1876], [13.4124, -0.7000], [13.6000, -1.4000],
            [13.7876, -2.1000], [14.3000, -2.6124], [15.0000, -2.8000],
            [16.0000, -2.8000]]
    return path


def test_course2():
    path = [[0, 0], [14.7800, 0], [15.6427, -0.3573], [16.0000, -1.2200],
            [16.0000, -4.7800], [15.6427, -5.6427], [14.7800, -6.0000],
            [1.2200, -6.0000], [0.3573, -6.3573], [0.0000, -7.2200],
            [0.0000, -10.7800], [0.3573, -11.6427], [1.2200, -12.0000],
            [14.7800, -12.0000], [15.6427, -12.3573], [16.0000, -13.2200],
            [16.0000, -16.7800], [15.6427, -17.6427], [14.7800, -18.0000],
            [-3.7800, -18.0000], [-4.6427, -17.6427], [-5.0000, -16.7800],
            [-5.0000, -1.2200], [-4.6427, -0.3573], [-3.7800, 0], [0, 0]]
    return path


def hard_course(resolution=200):
    h0 = 15
    h1 = 6
    h2 = 3
    h3 = 15
    h4 = 5
    h5 = 10
    h6 = 3
    h7 = 6
    h8 = 2
    h9 = np.pi * 8 / 2
    a1 = 3 / 2
    b1 = 2
    h10 = np.pi * 8 / 2
    a2 = 2
    b2 = 3 / 2

    c1 = 0
    c2 = 4 * np.pi
    c3 = c2 / (4 * np.pi) + h0
    c4 = c3 + h1
    c5 = c4 + h2
    c6 = c5 + h3
    c7 = c6 + h4
    c8 = c7 + h5
    c9 = c8 + h6
    c10 = c9 + h7
    c11 = c10 + h8
    c12 = c11 + h9
    c13 = c12 + h10

    s1 = c5 - c4 + c3
    s2 = c7 - c6 - s1
    s3 = -c7 - (c6 - c5 - h1)
    s4 = c8 + s3
    s5 = -(c9 - c8) - s2
    s6 = -(c10 - c9) + s4
    s7 = -(c11 - c10) + s5
    s9 = (-(c12 - c11) * np.cos(c12 - c11)) / a1 + s7
    s10 = ((c12 - c11) * np.sin(c12 - c11)) / b1 + s6

    s11 = (-(c13 - c12) * np.sin(c13 - c12)) / a2 + s9
    s12 = ((c13 - c12) * np.cos(c13 - c12)) / b2 + s10

    c14 = c13 + s11
    c15 = c14 + np.pi

    x = np.linspace(c1, c15, num=resolution)

    conditions = [
        np.logical_and(c1 <= x, x <= c2),
        np.logical_and(c2 <= x, x <= c3),
        np.logical_and(c3 <= x, x <= c4),
        np.logical_and(c4 <= x, x <= c5),
        np.logical_and(c5 <= x, x <= c6),
        np.logical_and(c6 <= x, x <= c7),
        np.logical_and(c7 <= x, x <= c8),
        np.logical_and(c8 <= x, x <= c9),
        np.logical_and(c9 <= x, x <= c10),
        np.logical_and(c10 <= x, x <= c11),
        np.logical_and(c11 <= x, x <= c12),
        np.logical_and(c12 <= x, x <= c13),
        np.logical_and(c13 <= x, x <= c14),
        np.logical_and(c14 <= x, x <= c15),
    ]

    x_eq = [
        lambda t: t,
        lambda t: t,
        lambda t: c3,
        lambda t: t - c4 + c3,
        lambda t: c5 - c4 + c3,
        lambda t: -(t - c6 - s1),
        lambda t: -s2,
        lambda t: -(t - c8) - s2,
        lambda t: s5,
        lambda t: -(t - c10) + s5,
        lambda t: (-(t - c11) * np.cos(t - c11)) / a1 + s7,
        lambda t: (-(t - c12) * np.sin(t - c12)) / a2 + s9,
        lambda t: -(t - c13) + s11,
        lambda t: -s12 / 2 * np.sin(t - c14),
    ]

    x_p = np.piecewise(x, conditions, x_eq)

    y_eq = [
        lambda t: 2 * (np.cos(t) - 1),
        lambda t: 0,
        lambda t: t - c3,
        lambda t: h1,
        lambda t: -(t - c5 - h1),
        lambda t: -(c6 - c5 - h1),
        lambda t: t + s3,
        lambda t: c8 + s3,
        lambda t: -(t - c9) + s4,
        lambda t: s6,
        lambda t: ((t - c11) * np.sin(t - c11)) / b1 + s6,
        lambda t: ((t - c12) * np.cos(t - c12)) / b2 + s10,
        lambda t: s12,
        lambda t: s12 / 2 + s12 / 2 * np.cos(t - c14),
    ]

    y_p = np.piecewise(x, conditions, y_eq)

    return np.vstack([x_p, y_p]).T.tolist()
