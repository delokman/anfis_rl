import tensorflow as tf


class ConsequentLayer(tf.Module):
    def __init__(self, mamdani_defs, output_membership_mapping) -> None:
        super().__init__()
        self.mamdani_defs = mamdani_defs
        self.output_membership_mapping = tf.constant(output_membership_mapping)

    def __call__(self):
        vs = self.mamdani_defs()

        return tf.expand_dims(tf.gather_nd(vs, self.output_membership_mapping), 1)
