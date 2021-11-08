import tensorflow as tf
from tensorflow import Variable


class JointFuzzifyLayer(tf.Module):
    def __init__(self, input_functions, name=None):
        super().__init__(name)

        with self.name_scope:
            self.input_functions = input_functions

        self.max_outputs = tf.constant(max(i.num_outputs for i in self.input_functions))

    @tf.Module.with_name_scope
    def __call__(self, x):
        i = tf.constant(0)

        output = tf.TensorArray(x.dtype, x.get_shape()[1])

        i0 = [i, output]

        c = lambda i, _: tf.less(i, len(self.input_functions))

        def foo(i, output):
            f = x[:, i:i+1]

            size = self.input_functions[i](f)
            output = output.write(i, size)
            return (i + 1, output)

        tf.while_loop(c, foo, i0)

        y_pred = tf.transpose(output.stack(), (1, 0, 2))

        print(y_pred)

        return y_pred
