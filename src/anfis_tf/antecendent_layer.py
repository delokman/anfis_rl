import tensorflow as tf


class AntecedentLayer(tf.Module):
    def __init__(self, mamdani_ruleset=None, name=None):
        super().__init__(name)

        if mamdani_ruleset is None:
            membership_indices = [[[[0, 0], [2, 0], [0, 0]], [[0, 0], [2, 1], [0, 0]], [[0, 0], [2, 2], [0, 0]],
                                   [[0, 1], [1, 0], [3, 0]],
                                   [[0, 1], [1, 0], [3, 1]], [[0, 1], [1, 0], [3, 2]], [[0, 1], [1, 0], [3, 3]],
                                   [[0, 1], [1, 0], [3, 4]],
                                   [[0, 1], [1, 1], [4, 0]], [[0, 1], [1, 1], [4, 1]], [[0, 1], [1, 1], [4, 2]],
                                   [[0, 1], [1, 1], [4, 3]],
                                   [[0, 1], [1, 1], [4, 4]], [[0, 1], [1, 2], [4, 0]], [[0, 1], [1, 2], [4, 1]],
                                   [[0, 1], [1, 2], [4, 2]],
                                   [[0, 1], [1, 2], [4, 3]], [[0, 1], [1, 2], [4, 4]], [[0, 1], [1, 3], [4, 0]],
                                   [[0, 1], [1, 3], [4, 1]],
                                   [[0, 1], [1, 3], [4, 2]], [[0, 1], [1, 3], [4, 3]], [[0, 1], [1, 3], [4, 4]],
                                   [[0, 1], [1, 4], [4, 0]],
                                   [[0, 1], [1, 4], [4, 1]], [[0, 1], [1, 4], [4, 2]], [[0, 1], [1, 4], [4, 3]],
                                   [[0, 1], [1, 4], [4, 4]],
                                   [[0, 1], [1, 5], [4, 0]], [[0, 1], [1, 5], [4, 1]], [[0, 1], [1, 5], [4, 2]],
                                   [[0, 1], [1, 5], [4, 3]],
                                   [[0, 1], [1, 5], [4, 4]], [[0, 1], [1, 6], [3, 0]], [[0, 1], [1, 6], [3, 1]],
                                   [[0, 1], [1, 6], [3, 2]],
                                   [[0, 1], [1, 6], [3, 3]], [[0, 1], [1, 6], [3, 4]], [[0, 0], [2, 2], [0, 0]],
                                   [[0, 0], [2, 3], [0, 0]],
                                   [[0, 0], [2, 4], [0, 0]]]]

            '''
            0 = far left
            1 = left
            2 = close left
            3 = zero
            4 = close right
            5 = right
            6 = far right
            '''
            outputs_membership = [
                (8,),  # 1
                (6,),  # 1
                (4,),  # 1
                (8,),  # 1
                (5,),  # 1
                (4,),  # 1
                (3,),  # 1
                (0,),  # 1

                (8,),  # 1
                (7,),  # 2
                (6,),  # 3
                (5,),  # 4
                (2,),  # 5
                (8,),  # 6
                (6,),  # 7
                (5,),  # 8
                (4,),  # 9
                (1,),  # 10
                (7,),  # 11
                (5,),  # 12
                (4,),  # 13
                (3,),  # 14
                (1,),  # 15

                # OTHER SIDE

                (7,),  # 1
                (4,),  # 2
                (3,),  # 3
                (2,),  # 4
                (0,),  # 5
                (6,),  # 6
                (3,),  # 7
                (2,),  # 8
                (1,),  # 9
                (0,),  # 10
                (8,),  # 11
                (5,),  # 12
                (4,),  # 13
                (3,),  # 14
                (0,),  # 15
                (4,),  # 15
                (2,),  # 15
                (0,),  # 15
            ]

            mamdani_ruleset = {
                'membership_indices': tf.constant(membership_indices),
                'outputs_membership': tf.constant(outputs_membership)
            }

        self.mamdani_ruleset = mamdani_ruleset

        self.index_cache = dict()

    def __call__(self, x):
        # x: 1, 5, 7

        # 1, 25, 2

        if x.shape[0] not in self.index_cache:
            # TODO improve implement in order to remove the use of repeat
            indices = tf.repeat(self.mamdani_ruleset['membership_indices'], x.shape[0], axis=0)
            self.index_cache[x.shape[0]] = indices
        else:
            indices = self.index_cache[x.shape[0]]

        weights = tf.gather_nd(x, indices, batch_dims=1)

        return tf.reduce_min(weights, axis=2)  # 1, 41


if __name__ == '__main__':
    a = 4
    x = tf.range(5 * 7 * a)
    x = tf.reshape(x, (a, 5, 7))
    a = AntecedentLayer()

    print(x)
    y = a(x)
    print(y)
