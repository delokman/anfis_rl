from abc import ABC, abstractmethod

import tensorflow as tf


class JointMembership(tf.Module, ABC):
    def __init__(self, num_outputs):
        super(JointMembership, self).__init__()

        self.num_outputs = tf.constant(num_outputs)
        self._padding = 0

        self.padding_array = tf.zeros(0)

    @property
    def padding(self):
        return self._padding

    @padding.setter
    def padding(self, max_outputs):
        self._padding = max_outputs - self.num_outputs

        if self.padding > 0:
            self.padding_array = tf.zeros(self.padding)

    @abstractmethod
    def compute(self, x):
        pass

    def __call__(self, x):
        y_pred = self.compute(x)
        y_pred = tf.clip_by_value(y_pred, 0, 1)

        return y_pred


class Test(JointMembership):
    def __init__(self):
        super().__init__(7)

    def compute(self, x):
        return tf.concat([x] * 7, axis=1)
