from anfis.consequent_layer import ConsequentLayerType
from anfis.joint_mamdani_membership import JointSymmetricTriangleMembership
from anfis.joint_membership_optimized import JointTrapMembershipV2
from anfis.trainer import make_joint_anfis


def predefined_anfis_model():
    parameter_values = [
        [0, 1.40812349319458, -0.06754400581121445, -0.06204153597354889],
        [0, 5.350396156311035, 0.152223140001297, 1.2252693176269531],
        [0, 2.367072820663452, 0.004285290837287903, 1.275625467300415],
        [0.5395, 0.525, 0.6303, 0.4556]
    ]

    x_joint_definitons = [
        ('distance', JointTrapMembershipV2(*parameter_values[0], constant_center=True)),
        ('theta_far', JointTrapMembershipV2(*parameter_values[1], constant_center=True)),
        ('theta_near', JointTrapMembershipV2(*parameter_values[2], constant_center=True)),
    ]

    output_names = ['angular_velocity']

    mambani = JointSymmetricTriangleMembership(*parameter_values[3], False,
                                               x_joint_definitons[0][1].required_dtype())

    rules_type = ConsequentLayerType.MAMDANI

    model = make_joint_anfis(x_joint_definitons, output_names, rules_type=rules_type, mamdani_defs=mambani)

    return model
