import numpy as np

from anfis.antecedent_layer import dist_target_dist_per_theta_lookahead_theta_far_theta_near, \
    dist_target_dist_per_theta_lookahead_theta_far_theta_near_with_vel
from anfis.consequent_layer import ConsequentLayerType
from anfis.joint_mamdani_membership import JointSymmetricTriangleMembership, JointSymmetric9TriangleMembership, \
    JointSymmetric3TriangleMembership
from anfis.joint_membership_hyperoptimized import JointSingleConstrainedEdgeMembershipV2, Joint7TrapMembershipV2, \
    JointTrapMembershipV3
from anfis.joint_membership_optimized import JointTrapMembershipV2, JointSingleConstrainedEdgeMembership, \
    Joint7TrapMembership
from anfis.trainer import make_joint_anfis
from new_ddpg.input_membership import JointTrapMembership
# from new_ddpg.new_anfis import JointAnfisNet
from new_ddpg.new_anfis import min_max_num_trapezoids, min_max_num_symmetric_center_of_max, min_max_num_center_of_max, \
    JointAnfisNet
from new_ddpg.output_membership import SymmetricCenterOfMaximum, CenterOfMaximum


def predefined_anfis_model():
    parameter_values = [
        [0, 1.40812349319458, .1, 0.699826002120972],
        [0, 0.976657846180786, 0.27020001411438, 0.1281498670578],
        # [0, 1.12320299803035, 0.081358410418034, 0.103709816932678],
        [0, 1.52320299803035, 0.081358410418034, 0.103709816932678],  # Old theta near
        # [0, 4.666666666666667, 0.21428571428571427, 0.21428571428571427],
        # [0.5395, 0.525, 0.6303, 0.4556]
        [0, 1.8763, 1.9412, 2.1448]
    ]

    x_joint_definitons = [
        ('distance', JointTrapMembershipV2(*parameter_values[0], constant_center=True)),
        ('theta_far', JointTrapMembershipV2(*parameter_values[1], constant_center=True)),
        ('theta_near', JointTrapMembershipV2(*parameter_values[2], constant_center=True)),
    ]

    output_names = ['angular_velocity']

    # mambani = JointSymmetricTriangleMembership(*parameter_values[3], False,
    #                                            x_joint_definitons[0][1].required_dtype())
    mambani = JointSymmetricTriangleMembership(*parameter_values[3], True,
                                               x_joint_definitons[0][1].required_dtype())

    rules_type = ConsequentLayerType.MAMDANI

    model = make_joint_anfis(x_joint_definitons, output_names, rules_type=rules_type, mamdani_defs=mambani)

    return model


def many_error_predefined_anfis_model():
    parameter_values = [
        [0, 1],

        [0, 2, .1, .2, .2],
        [0, 1, .6, 0.6],
        [0, 0.976657846180786, 0.27020001411438, 0.1281498670578],
        [0, 1.52320299803035, 0.081358410418034, 0.103709816932678],  #

        [0, 1, 1, 1, 1]
    ]

    x_joint_definitons = [
        ('distance_target', JointSingleConstrainedEdgeMembership(*parameter_values[0], constant_center=False)),

        ('distance_line', Joint7TrapMembership(*parameter_values[1], constant_center=True)),
        ('theta_lookahead', JointTrapMembershipV2(*parameter_values[2], constant_center=True)),
        ('theta_far', JointTrapMembershipV2(*parameter_values[3], constant_center=True)),
        ('theta_near', JointTrapMembershipV2(*parameter_values[4], constant_center=True)),
    ]

    output_names = ['angular_velocity']

    mambani = JointSymmetric9TriangleMembership(*parameter_values[5], True,
                                                x_joint_definitons[0][1].required_dtype())

    rules_type = ConsequentLayerType.MAMDANI

    ruleset = dist_target_dist_per_theta_lookahead_theta_far_theta_near()

    model = make_joint_anfis(x_joint_definitons, output_names, rules_type=rules_type, mamdani_defs=mambani,
                             mamdani_ruleset=ruleset)

    return model


def optimized_many_error_predefined_anfis_model():
    parameter_values = [
        [0.001, 1.],

        [0., 2., .1, .2, .2],
        [0., 1., .6, 0.6],
        [0., 0.976657846180786, 0.27020001411438, 0.1281498670578],
        [0., 1.52320299803035, 0.081358410418034, 0.103709816932678],  #

        [0., 1., 1., 1., 1.]
    ]

    x_joint_definitons = [
        ('distance_target', JointSingleConstrainedEdgeMembershipV2(*parameter_values[0], constant_center=False)),

        ('distance_line', Joint7TrapMembershipV2(*parameter_values[1], constant_center=True)),
        ('theta_lookahead', JointTrapMembershipV3(*parameter_values[2], constant_center=True)),
        ('theta_far', JointTrapMembershipV3(*parameter_values[3], constant_center=True)),
        ('theta_near', JointTrapMembershipV3(*parameter_values[4], constant_center=True)),
    ]

    output_names = ['angular_velocity']

    mambani = JointSymmetric9TriangleMembership(*parameter_values[5], True,
                                                x_joint_definitons[0][1].required_dtype())

    rules_type = ConsequentLayerType.MAMDANI

    ruleset = dist_target_dist_per_theta_lookahead_theta_far_theta_near()

    model = make_joint_anfis(x_joint_definitons, output_names, rules_type=rules_type, mamdani_defs=mambani,
                             mamdani_ruleset=ruleset)

    return model


def optimized_many_error_predefined_anfis_model_with_velocity():
    parameter_values = [
        [0.001, 1.],

        [0., 2., .1, .2, .2],
        [0., 1., .6, 0.6],
        [0., 1, .25, 0.125],
        [0., 1.5, 0.1, 0.1],  #

        [0., 1., 1., 1., 1.],

        # [0., 3.146184206008911, 7.892924622865394e-05, -0.052046310156583786, 0.19876137375831604],
        # [0., 1.4081424474716187, 0.3765484690666199, 0.24828855693340302],
        # [0., 0.9766578674316406, 0.2702000141143799, 0.1281498670578003],
        # [0., 2.227175235748291, 0.0006221223738975823, 0.004938468802720308],  #
        #
        # [0., 1.9652783870697021, 1.919374704360962, 1.4656635522842407, 1.4181907176971436],
        # [0.0017236630665138364, 1.2268195152282715],

        # [0.2, .8, 1.]
        [0.2, .3, .8]
        # [0.2, 1, 2.]
    ]

    x_joint_definitons = [
        ('distance_target', JointSingleConstrainedEdgeMembershipV2(*parameter_values[0], constant_center=False)),

        ('distance_line', Joint7TrapMembershipV2(*parameter_values[1], constant_center=True)),
        ('theta_lookahead', JointTrapMembershipV3(*parameter_values[2], constant_center=True)),
        ('theta_far', JointTrapMembershipV3(*parameter_values[3], constant_center=True)),
        ('theta_near', JointTrapMembershipV3(*parameter_values[4], constant_center=True)),
    ]

    output_names = ['angular_velocity', 'velocity']

    mambani = JointSymmetric9TriangleMembership(*parameter_values[5], True,
                                                x_joint_definitons[0][1].required_dtype())

    mambani_vel = JointSymmetric3TriangleMembership(*parameter_values[6], x_joint_definitons[0][1].required_dtype())

    rules_type = ConsequentLayerType.MAMDANI

    ruleset = dist_target_dist_per_theta_lookahead_theta_far_theta_near_with_vel()

    model = make_joint_anfis(x_joint_definitons, output_names, rules_type=rules_type,
                             mamdani_defs=[mambani, mambani_vel],
                             mamdani_ruleset=ruleset, velocity=True)

    return model


def new_anfis_many_error_with_velocity():
    mem1 = JointTrapMembership(*min_max_num_trapezoids(0, 1, 2))
    mem2 = JointTrapMembership(*min_max_num_trapezoids(-1.95, 1.95, 7))
    mem3 = JointTrapMembership(*min_max_num_trapezoids(-2.9, 2.9, 5))
    mem4 = JointTrapMembership(*min_max_num_trapezoids(-2.3, 2.3, 5))
    mem5 = JointTrapMembership(*min_max_num_trapezoids(-1.5, 1.5, 5))

    ruleset = dist_target_dist_per_theta_lookahead_theta_far_theta_near_with_vel()

    out1 = SymmetricCenterOfMaximum(*min_max_num_symmetric_center_of_max(-4, 4, 9))
    out2 = CenterOfMaximum(min_max_num_center_of_max(.2, 2, 3))

    anfis = JointAnfisNet([mem1, mem2, mem3, mem4, mem5], [out1, out2], ruleset, [0, 0], [4, 2])

    return anfis
