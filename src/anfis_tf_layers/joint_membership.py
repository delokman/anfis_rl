from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import keras


class JointMembership(keras.layers.Layer, ABC):
    padding_cache = dict()

    def __init__(self, num_outputs):
        super(JointMembership, self).__init__()
        self.num_outputs = tf.constant(num_outputs)

        self.padding = 0

        self.padding_c = tf.zeros(0)

    def pad_to(self, padding):
        v = padding - self.num_outputs

        if v > 0:
            self.padding = v

    def build(self, input_shape):
        print(len(JointMembership.padding_cache))
        if self.padding > 0:
            if self.padding.ref() not in JointMembership.padding_cache:
                self.padding_c = tf.zeros((input_shape[0], self.padding))
                JointMembership.padding_cache[self.padding.ref()] = self.padding_c
            else:
                self.padding_c = JointMembership.padding_cache[self.padding.ref()]


    @abstractmethod
    def compute(self, x):
        pass

    def call(self, x):
        x = self.compute(x)
        x = tf.clip_by_value(x, 0, 1)

        if self.padding > 0:
            x = tf.concat([x, self.padding_c], axis=1)
        return x


class Test(JointMembership):
    def __init__(self):
        super().__init__(7)

    def compute(self, x):
        return tf.repeat(x, 7, axis=1)
