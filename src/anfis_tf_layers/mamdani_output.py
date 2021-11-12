import tensorflow as tf
from tensorflow.keras.layers import Layer


class JointSymmetric9TriangleMembership(Layer):

    def __getitem__(self, item):
        return tf.switch_case(item, self.output_function)

    def get_center(self):
        return self.center

    def get_soft(self, direction=1):
        return self.get_center() + direction * tf.abs(self.soft)

    def get_normal(self, direction=1):
        return self.get_soft(direction=direction) + direction * tf.abs(self.normal)

    def get_hard(self, direction=1):
        return self.get_normal(direction=direction) + direction * tf.abs(self.hard)

    def get_very_hard(self, direction=1):
        return self.get_hard(direction=direction) + direction * tf.abs(self.very_hard)

    def __init__(self, center, soft, normal, hard, very_hard, constant_center=True) -> None:
        super().__init__()

        self.center = tf.Variable(center, trainable=constant_center)

        self.soft = tf.Variable(soft)
        self.normal = tf.Variable(normal)
        self.hard = tf.Variable(hard)
        self.very_hard = tf.Variable(very_hard)

        # self.output_function = {
        #     0: partial(self.get_very_hard, direction=1),
        #     1: partial(self.get_hard, direction=1),
        #     2: partial(self.get_normal, direction=1),
        #     3: partial(self.get_soft, direction=1),
        #     4: self.get_center,
        #     5: partial(self.get_soft, direction=-1),
        #     6: partial(self.get_normal, direction=-1),
        #     7: partial(self.get_hard, direction=-1),
        #     8: partial(self.get_very_hard, direction=-1),
        # }

    def call(self, x, **kwargs):
        llll = self.get_very_hard(1)
        lll = self.get_hard(1)
        ll = self.get_normal(1)
        l = self.get_soft(1)
        c = self.get_center()
        r = self.get_soft(-1)
        rr = self.get_normal(-1)
        rrr = self.get_hard(-1)
        rrrr = self.get_very_hard(-1)

        return tf.stack([llll, lll, ll, l, c, r, rr, rrr, rrrr])
