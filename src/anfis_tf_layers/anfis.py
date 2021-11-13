from datetime import datetime
from typing import List

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

tf.config.optimizer.set_jit(True)

from anfis_tf_layers.joint_membership import JointMembership, JointTrap5Membership, JointTrap7Membership, \
    JointSingleConstrainedEdgeMembership
from anfis_tf_layers.mamdani_output import JointSymmetric9TriangleMembership
from anfis_tf_layers.rules import rules


class JointFuzzifyLayer(Layer):
    def __init__(self, input_layers: List[JointMembership]):
        super(JointFuzzifyLayer, self).__init__()

        self.input_layers = input_layers
        self.max_outputs = tf.constant(max([i.num_outputs for i in input_layers]))

        for i in input_layers:
            i.pad_to(self.max_outputs)

    def call(self, x, **kwargs):
        ouptuts = []

        for i, inp in enumerate(self.input_layers):
            ouptuts.append(inp(x[:, i:i + 1]))

        o = tf.stack(ouptuts, axis=1)
        return o


class AntecedentLayer(Layer):
    def __init__(self, indexes):
        super(AntecedentLayer, self).__init__()

        self._indexes = tf.constant(indexes)
        self.indices = self._indexes

    def build(self, input_shape):
        self.indices = tf.repeat(self._indexes, input_shape[0], axis=0)

    def call(self, x, **kwargs):
        weights = tf.gather_nd(x, self.indices, batch_dims=1)

        return tf.reduce_min(weights, axis=2)


class ConsequentLayer(Layer):
    def __init__(self, output_membership_mapping) -> None:
        super().__init__()
        self.output_membership_mapping = tf.constant(output_membership_mapping)

    def call(self, mamdani_output, **kwargs):
        return tf.expand_dims(tf.gather_nd(mamdani_output, self.output_membership_mapping), 1)


class Normalize(Layer):
    def call(self, x, **kwargs):
        x, _ = tf.linalg.normalize(x, ord=1, axis=0)
        return x


class Multiply(Layer):
    def call(self, inputs, **kwargs):
        x, rules = inputs
        return tf.matmul(x, rules)


class ANFIS(Model):
    def __init__(self, input_layers, output_functions, mamdani_ruleset):
        super(ANFIS, self).__init__()

        self.fuzzify = JointFuzzifyLayer(input_layers)
        self.rules = AntecedentLayer(mamdani_ruleset['membership_indices'])
        self.normalize = Normalize()
        self.output_function = output_functions
        self.consequent = ConsequentLayer(mamdani_ruleset['outputs_membership'])
        self.multiply = Multiply()

    def call(self, x, **kwargs):
        print(x.shape)
        x = self.fuzzify(x)
        x = self.rules(x)

        x = self.normalize(x)

        output = self.output_function([])

        rule_tsk = self.consequent(output)

        return self.multiply((x, rule_tsk))


if __name__ == '__main__':
    from keras import backend as K
    print(K.tensorflow_backend._get_available_gpus()  )

    n = 10000

    x = tf.random.uniform((n, 5), 0., 1., seed=42)
    y = tf.random.uniform((n,), 0., 1., seed=42)

    joint_definitons = [JointSingleConstrainedEdgeMembership(0.001, 1., constant_center=False),
                        JointTrap7Membership(0., 2., .1, .2, .2, constant_center=True),
                        JointTrap5Membership(0., 1., .6, .6, constant_center=True),
                        JointTrap5Membership(0., 1., .3, .1, constant_center=True),
                        JointTrap5Membership(0., 1.5, 0.1, 0.1, constant_center=True),
                        ]

    model = ANFIS(joint_definitons, JointSymmetric9TriangleMembership(0., 1., 1., 1., 1.),
                  rules())
    model.compile('sgd', 'mse')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", write_graph=True, profile_batch=1)
    data = tf.data.Dataset.from_tensors((x, y))
    data.batch(64)

    model.fit(data,  callbacks=[tensorboard_callback], epochs=10)

    print(model.summary())
