from abc import ABC, abstractmethod

import tensorflow as tf
from keras.layers import Concatenate
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tensorflow.keras.layers import Layer


class JointMembership(Layer, ABC):
    padding_cache = dict()

    @abstractmethod
    def left_x(self):
        pass

    @abstractmethod
    def half_width(self):
        pass

    @abstractmethod
    def right_x(self):
        pass

    def __init__(self, num_outputs, membership_names, name=None):
        super(JointMembership, self).__init__(name=name)
        self.membership_names = membership_names
        self.num_outputs = tf.constant(num_outputs)

        self.padding = tf.constant(0)

        self.padding_c = tf.zeros(0)
        self.concat = Concatenate(axis=1)

    def pad_to(self, padding):
        v = padding - self.num_outputs

        if v > 0:
            self.padding = tf.constant(v)

    def build(self, input_shape):
        if self.padding > 0:
            if self.padding.ref() not in JointMembership.padding_cache:
                self.padding_c = tf.zeros((input_shape[0], self.padding))
                JointMembership.padding_cache[self.padding.ref()] = self.padding_c
            else:
                self.padding_c = JointMembership.padding_cache[self.padding.ref()]

    @abstractmethod
    def compute(self, x):
        pass

    def call(self, x, **kwargs):
        x = self.compute(x)
        x = tf.clip_by_value(x, tf.keras.backend.epsilon(), 1.)

        # if tf.greater(self.padding, 0):
        if self.padding_c.shape[0] > 0:
            # print(self.name,self.padding > 0)
            # print(self.padding_c.shape)
            x = tf.pad(x, [[0, 0], [0, self.padding]])
            # x = tf.concat([x, self.padding_c], axis=1)
            # x = self.concat([x, self.padding_c])
        return x

    def plot(self, figure: Figure, n=500):
        offset = self.half_width() * .05

        x = tf.linspace(self.left_x() - offset, self.right_x() + offset, n)
        y = self.call(tf.expand_dims(x, 1))

        ax: Axes = figure.gca()

        for i, name in enumerate(self.membership_names):
            ax.plot(x, y[:, i], label=name)

        ax.set_title(f'{self.name}')
        ax.legend()


class JointTrap5Membership(JointMembership):
    def left_x(self):
        return self.center - self.half_width()

    def right_x(self):
        return self.center + self.half_width()

    def half_width(self):
        return tf.abs(self.center_width) / 2 + tf.abs(self.side_width) + 2 / tf.abs(
            self.slope - self.slope_constraint) + self.slope_constraint

    def __init__(self, center, slope, center_width, side_width, constant_center=False, min_slope=0.01, name=None):
        super().__init__(5, ["Left Edge", "Left", "Center", "Right", "Right Edge"], name=name)
        self.slope_constraint = tf.constant(min_slope)

        with tf.name_scope(self.name):
            self.center = tf.Variable(center, trainable=not constant_center, name='center')
            self.slope = tf.Variable(slope, name='slope')
            self.center_width = tf.Variable(center_width, name='center_width')
            self.side_width = tf.Variable(side_width, name='side_width')

    def compute(self, x):
        slope = tf.abs(self.slope - self.slope_constraint) + self.slope_constraint

        slope_width = 1 / slope

        center_width = tf.abs(self.center_width)
        side_width = tf.abs(self.side_width)

        x = x - self.center

        center_div_2 = center_width / 2
        side_div_2 = side_width / 2

        # CENTER

        x_center = tf.abs(x)
        center = - slope * (x_center - center_div_2) + 1

        # CLOSE LEFT AND RIGHT

        shift = slope * side_div_2 + 1
        c = center_div_2 + slope_width + side_div_2

        # LEFT

        x_close_left = tf.abs(x + c)
        close_left = -slope * x_close_left + shift

        # RIGHT
        x_close_right = tf.abs(x - c)
        close_right = -slope * x_close_right + shift

        # EDGES

        c = center_div_2 + side_width + 2 * slope_width

        # LEFT EDGE
        x_left_edge = -x - c
        left_edge = slope * x_left_edge + 1

        # RIGHT EDGE
        x_right_edge = x - c
        right_edge = slope * x_right_edge + 1

        return tf.concat([left_edge, close_left, center, close_right, right_edge], axis=1)


class JointSingleConstrainedEdgeMembership(JointMembership):
    def left_x(self):
        return tf.abs(self.center)

    def half_width(self):
        return 1 / (2 * tf.abs(self.slope - self.slope_constraint) + self.slope_constraint)

    def right_x(self):
        return self.left_x() + self.half_width() * 2

    def __init__(self, center, slope, constant_center=False, min_slope=0.01, name=None):
        super().__init__(2, ['Close', "Far"], name=name)
        self.slope_constraint = tf.constant(min_slope)

        with tf.name_scope(self.name):
            self.center = tf.Variable(center, trainable=not constant_center, name='center')
            self.slope = tf.Variable(slope, name='slope')

    def compute(self, x):
        slope = tf.abs(self.slope - self.slope_constraint) + self.slope_constraint
        center = tf.abs(self.center)

        x = x - center

        right = slope * x
        left = 1 - right

        return tf.concat([left, right], axis=1)


class JointTrap7Membership(JointMembership):
    def left_x(self):
        return self.center - self.half_width()

    def half_width(self):
        return tf.abs(self.center_width) / 2 + tf.abs(self.side_width) + tf.abs(
            self.super_side_width) + 3 / (tf.abs(self.slope - self.slope_constraint) + self.slope_constraint)

    def right_x(self):
        return self.center + self.half_width()

    def __init__(self, center, slope, center_width, side_width, super_side_width, constant_center=False,
                 min_slope=0.01, name=None):
        super().__init__(7, ['Left Edge', 'Left', 'Close Left', "Center", 'Close Right', "Right", 'Right Edge'],
                         name=name)
        self.slope_constraint = tf.constant(min_slope)

        with tf.name_scope(self.name):
            self.center = tf.Variable(center, trainable=not constant_center, name='center')
            self.slope = tf.Variable(slope, name='slope')
            self.center_width = tf.Variable(center_width, name='center_width')
            self.side_width = tf.Variable(side_width, name='side_width')
            self.super_side_width = tf.Variable(super_side_width, name='super_side_width')

    def compute(self, x):
        # IMPLEMENT TECHNIQUE SO THAT THE OUTPUT MATRIX IS FIST ALL ZEROS.
        # When the area is 1, for the other membership functions it is 0

        slope = tf.abs(self.slope - self.slope_constraint) + self.slope_constraint

        slope_width = 1 / slope

        center_width = tf.abs(self.center_width)
        side_width = tf.abs(self.side_width)
        super_side_width = tf.abs(self.super_side_width)

        x = x - self.center

        center_div_2 = center_width / 2
        side_div_2 = side_width / 2
        super_side_div_2 = super_side_width / 2

        # CENTER

        x_center = tf.abs(x)
        center = - slope * (x_center - center_div_2) + 1

        # CLOSE LEFT AND RIGHT

        shift = slope * side_div_2 + 1
        c = center_div_2 + slope_width + side_div_2

        # LEFT

        x_close_left = tf.abs(x + c)
        close_left = -slope * x_close_left + shift

        # RIGHT
        x_close_right = tf.abs(x - c)
        close_right = -slope * x_close_right + shift

        # LEFT AND RIGHT

        shift = slope * super_side_div_2 + 1
        c = center_div_2 + 2 * slope_width + side_width + super_side_div_2

        # LEFT

        x_left = tf.abs(x + c)
        left = -slope * x_left + shift

        # RIGHT

        x_right = tf.abs(x - c)
        right = -slope * x_right + shift

        # EDGES

        c = center_div_2 + side_width + 3 * slope_width + super_side_width

        # LEFT EDGE
        x_left_edge = -x - c
        left_edge = slope * x_left_edge + 1

        # RIGHT EDGE
        x_right_edge = x - c
        right_edge = slope * x_right_edge + 1

        return tf.concat([left_edge, left, close_left, center, close_right, right, right_edge], axis=1)
