import tensorflow as tf
from tensorflow import Variable


class JointFuzzifyLayer(tf.Module):
    def __init__(self, input_functions, name=None):
        super().__init__(name)

        with self.name_scope:
            self.input_functions = [tf.function(i) for i in input_functions]

        self.max_outputs = tf.constant(max(i.num_outputs for i in self.input_functions))
        self.num_lingustic_variables = tf.constant(len(self.input_functions))

        for func in self.input_functions:
            func.pad_to(self.max_outputs)

    @tf.Module.with_name_scope
    def __call__(self, x):
        i = tf.constant(0)

        output = tf.TensorArray(x.dtype, x.shape[1])

        i0 = [i, output]

        c = lambda i, _: tf.less(i, self.num_lingustic_variables)

        # FIXME make this eager executable
        def foo(i, output):
            f = x[:, i:i + 1]

            val = self.input_functions[i](f)
            output = output.write(i, val)
            return (i + 1, output)

        tf.while_loop(c, foo, i0)

        y_pred = tf.transpose(output.stack(), (1, 0, 2))

        return y_pred
