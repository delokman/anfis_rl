import tensorflow as tf

from anfis_tf.consequent_layer import ConsequentLayer
from anfis_tf.antecendent_layer import AntecedentLayer
from anfis_tf.fuzzy_layer import JointFuzzifyLayer
from anfis_tf.joint_mamdani_membership import JointSymmetric9TriangleMembership
from anfis_tf.joint_membership import Test


class MamadaniANFIS(tf.Module):
    def __init__(self, input_functions, output_functions, mamdani_ruleset=None, name=None):
        super().__init__(name=name)

        with self.name_scope:
            self.input_functions = input_functions
            self.output_functions = output_functions

        self.fuzzify = JointFuzzifyLayer(input_functions)
        self.rules = AntecedentLayer(mamdani_ruleset)
        self.consequent = ConsequentLayer(output_functions, self.rules.mamdani_ruleset['outputs_membership'])

        self.rules = tf.function(self.rules)
        self.consequent = tf.function(self.consequent)

    @tf.Module.with_name_scope
    def __call__(self, x):
        fuzzify = self.fuzzify(x)
        raw_weights = self.rules(fuzzify)

        weights = tf.norm(raw_weights, ord=1, axis=1)

        rule_tsk = self.consequent()

        # Product-sum
        return tf.matmul(weights, rule_tsk)


if __name__ == '__main__':
    model = MamadaniANFIS([Test()], None)
    # model = tf.function(model)

    x1 = tf.linspace(-10, 10, 20)
    x2 = tf.linspace(10, -10, 20)

    x = tf.concat([x1, x2], axis=0)
    x = tf.reshape(x, (20, 2))

    model(x)
