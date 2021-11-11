from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import keras


class JointMembership(keras.layers.Layer, ABC):
    def __init__(self, num_outputs):
        super(JointMembership, self).__init__()
        self.num_outputs = tf.constant(num_outputs)

        self.padding = 0

        self.padding_c = tf.zeros(0)

    def pad_to(self, padding):
        v = padding - self.num_outputs

        if v > 0:
            self.padding = v
            self.padding_c = tf.zeros(v)

    @abstractmethod
    def compute(self, x):
        pass

    def call(self, x):
        x = self.compute(x)
        x = tf.clip_by_value(x, 0, 1)

        if self.padding > 0:
            a = tf.repeat(self.padding_c, x.shape[0])
            x = tf.concat([x, a], axis=0)
        return x


class Test(JointMembership):
    def __init__(self):
        super().__init__(7)

    def compute(self, x):
        return tf.repeat(x, 7, axis=1)
