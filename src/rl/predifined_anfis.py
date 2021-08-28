from anfis.consequent_layer import ConsequentLayerType
from anfis.joint_mamdani_membership import JointSymmetricTriangleMembership
from anfis.joint_membership_optimized import JointTrapMembershipV2
from anfis.trainer import make_joint_anfis


def predefined_anfis_model():
    parameter_values = [
        [0, 1.40812349319458, .1, 0.699826002120972],
        [0, 0.976657846180786, 0.27020001411438, 0.1281498670578],
        # [0, 1.12320299803035, 0.081358410418034, 0.103709816932678],
        [0, 1.52320299803035, 0.081358410418034, 0.103709816932678], #Old theta near
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
