import sys
import time

import matplotlib.pyplot as plt
import torch

from anfis.joint_membership_hyperoptimized import JointTrapMembershipV3, JointSingleConstrainedEdgeMembershipV2, \
    Joint7TrapMembershipV2
from anfis.joint_membership_matrix_hyperoptimized import JointTrapMembershipV4, JointSingleConstrainedEdgeMembershipV3, \
    Joint7TrapMembershipV3
from anfis.joint_membership_optimized import JointTrapMembershipV2, JointSingleConstrainedEdgeMembership, \
    Joint7TrapMembership
from anfis.matrix_fuzzy_layer import Matrix5ErrorJointFuzzifyLayer


def test_membership(funcs):
    with torch.no_grad():
        fv = funcs[0]

        membership_range = fv.right_x() - fv.left_x()
        offset = membership_range * .2

        left = fv.left_x() - offset
        right = fv.right_x() + offset

        x = torch.linspace(left, right, steps=32, dtype=fv.required_dtype(), device=left.device).unsqueeze(1)

        start = time.time()
        y_true = fv(x)
        end = time.time()
        print("Default: ", end - start)

        fig, ax = plt.subplots()

        # ax.plot(x.cpu(), y_true.cpu())

        fig, ax = plt.subplots()
        for i in range(1, len(funcs)):
            fv_i = funcs[i]
            start = time.time()
            y_i = fv_i(x)
            end = time.time()

            diff = y_true - y_i

            # ax.plot(x.cpu(), y_i.cpu())

            print(end - start, torch.abs(diff).sum().item())

        # plt.show()


def cpu_run():
    print("USING CPU")
    fv = JointTrapMembershipV2(1, 1, 1, 1)
    fv2 = JointTrapMembershipV3(1, 1, 1, 1)
    fv3 = JointTrapMembershipV4(1, 1, 1, 1)

    a1 = [fv, fv2, fv3]

    fv = JointSingleConstrainedEdgeMembership(1, 1)
    fv2 = JointSingleConstrainedEdgeMembershipV2(1, 1)
    fv3 = JointSingleConstrainedEdgeMembershipV3(1, 1)

    a2 = [fv, fv2, fv3]

    fv = Joint7TrapMembership(1, 1, 1, 1, 1)
    fv2 = Joint7TrapMembershipV2(1, 1, 1, 1, 1)
    fv3 = Joint7TrapMembershipV3(1, 1, 1, 1, 1)

    a3 = [fv, fv2, fv3]

    names = ["Trap 5", "Edge", "Trap 7"]

    for name, l in zip(names, [a1, a2, a3]):
        print(name)
        test_membership(l)


def gpu_run():
    print("USING GPU")
    fv = JointTrapMembershipV2(1, 1, 1, 1).cuda()
    fv2 = JointTrapMembershipV3(1, 1, 1, 1).cuda()
    fv3 = JointTrapMembershipV4(1, 1, 1, 1).cuda()

    a1 = [fv, fv2, fv3]

    fv = JointSingleConstrainedEdgeMembership(1, 1).cuda()
    fv2 = JointSingleConstrainedEdgeMembershipV2(1, 1).cuda()
    fv3 = JointSingleConstrainedEdgeMembershipV3(1, 1).cuda()

    a2 = [fv, fv2, fv3]

    fv = Joint7TrapMembership(1, 1, 1, 1, 1).cuda()
    fv2 = Joint7TrapMembershipV2(1, 1, 1, 1, 1).cuda()
    fv3 = Joint7TrapMembershipV3(1, 1, 1, 1, 1).cuda()

    a3 = [fv, fv2, fv3]

    names = ["Trap 5", "Edge", "Trap 7"]

    for name, l in zip(names, [a1, a2, a3]):
        print(name)
        test_membership(l)


if __name__ == '__main__':
    cpu_run()
    gpu_run()
    # plt.show()
