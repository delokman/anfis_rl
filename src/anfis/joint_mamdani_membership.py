from abc import ABC, abstractmethod
from functools import partial

import torch

from anfis.joint_membership import _mk_param


class JointMamdaniMembership(torch.nn.Module, ABC):
    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def cache(self):
        pass

    @abstractmethod
    def release_cache(self):
        pass

    @abstractmethod
    def to_cuda(self):
        pass


class JointSymmetricTriangleMembership(JointMamdaniMembership):

    def __getitem__(self, item):
        return self.cache_output_values[item]

    def cache(self):
        self.abs_cache['center'] = self.center
        self.abs_cache['soft'] = torch.abs(self.soft)
        self.abs_cache['normal'] = torch.abs(self.normal)
        self.abs_cache['hard'] = torch.abs(self.hard)

        for key, val in self.output_function.items():
            self.cache_output_values[key] = val()

    def release_cache(self):
        self.abs_cache.clear()
        self.cache_output_values.clear()

    def get_center(self):
        return self.abs_cache['center']

    def get_soft(self, direction=1):
        return self.abs_cache['center'] + direction * self.abs_cache['soft']

    def get_normal(self, direction=1):
        return self.abs_cache['center'] + direction * (self.abs_cache['soft'] + self.abs_cache['normal'])

    def get_hard(self, direction=1):
        return self.center + direction * (self.abs_cache['soft'] + self.abs_cache['normal'] + self.abs_cache['hard'])

    def to_cuda(self):
        self.center = self.center.to(device='cuda')

    def __init__(self, center, soft, normal, hard, constant_center=True, dtype=torch.float) -> None:
        super().__init__()

        if constant_center:
            self.center = torch.tensor(center, dtype=dtype, requires_grad=False)
        else:
            self.register_parameter('center', _mk_param(center, dtype=dtype))

        self.register_parameter('soft', _mk_param(soft, dtype=dtype))
        self.register_parameter('normal', _mk_param(normal, dtype=dtype))
        self.register_parameter('hard', _mk_param(hard, dtype=dtype))

        self.abs_cache = dict()

        self.output_function = {
            0: partial(self.get_hard, direction=1),
            1: partial(self.get_normal, direction=1),
            2: partial(self.get_soft, direction=1),
            3: self.get_center,
            4: partial(self.get_soft, direction=-1),
            5: partial(self.get_normal, direction=-1),
            6: partial(self.get_hard, direction=-1),
        }

        self.names = {
            0: 'Hard Left',
            1: 'Left',
            2: 'Soft Left',
            3: 'Zero',
            4: 'Soft Right',
            5: 'Right',
            6: 'Hard Right',
        }

        self.cache_output_values = dict()


class JointSymmetric9TriangleMembership(JointMamdaniMembership):

    def __getitem__(self, item):
        return self.cache_output_values[item]

    def cache(self):
        self.abs_cache['center'] = self.center
        self.abs_cache['soft'] = torch.abs(self.soft)
        self.abs_cache['normal'] = torch.abs(self.normal)
        self.abs_cache['hard'] = torch.abs(self.hard)
        self.abs_cache['very_hard'] = torch.abs(self.very_hard)

        for key, val in self.output_function.items():
            self.cache_output_values[key] = val()

    def release_cache(self):
        self.abs_cache.clear()
        self.cache_output_values.clear()

    def get_center(self):
        return self.abs_cache['center']

    def get_soft(self, direction=1):
        return self.abs_cache['center'] + direction * self.abs_cache['soft']

    def get_normal(self, direction=1):
        return self.abs_cache['center'] + direction * (self.abs_cache['soft'] + self.abs_cache['normal'])

    def get_hard(self, direction=1):
        return self.center + direction * (self.abs_cache['soft'] + self.abs_cache['normal'] + self.abs_cache['hard'])

    def get_very_hard(self, direction=1):
        return self.center + direction * (
                self.abs_cache['soft'] + self.abs_cache['normal'] + self.abs_cache['hard'] +
                self.abs_cache['very_hard'])

    def to_cuda(self):
        self.center = self.center.to(device='cuda')

    def __init__(self, center, soft, normal, hard, very_hard, constant_center=True, dtype=torch.float) -> None:
        super().__init__()

        if constant_center:
            self.center = torch.tensor(center, dtype=dtype, requires_grad=False)
        else:
            self.register_parameter('center', _mk_param(center, dtype=dtype))

        self.register_parameter('soft', _mk_param(soft, dtype=dtype))
        self.register_parameter('normal', _mk_param(normal, dtype=dtype))
        self.register_parameter('hard', _mk_param(hard, dtype=dtype))
        self.register_parameter('very_hard', _mk_param(very_hard, dtype=dtype))

        self.abs_cache = dict()

        self.output_function = {
            0: partial(self.get_very_hard, direction=1),
            1: partial(self.get_hard, direction=1),
            2: partial(self.get_normal, direction=1),
            3: partial(self.get_soft, direction=1),
            4: self.get_center,
            5: partial(self.get_soft, direction=-1),
            6: partial(self.get_normal, direction=-1),
            7: partial(self.get_hard, direction=-1),
            8: partial(self.get_very_hard, direction=-1),
        }

        self.names = {
            0: 'Very Hard Left',
            1: 'Hard Left',
            2: 'Left',
            3: 'Soft Left',
            4: 'Zero',
            5: 'Soft Right',
            6: 'Right',
            7: 'Hard Right',
            8: 'Very Hard Right',
        }

        self.cache_output_values = dict()
