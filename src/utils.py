import numpy as np
from torch.utils.tensorboard.summary import hparams


def angdiff(th1, th2):
    d = th1 - th2
    d = np.mod(d + np.pi, 2 * np.pi) - np.pi
    return -d


def wraptopi(x):
    pi = np.pi
    x = x - np.floor(x / (2 * pi)) * 2 * pi
    if x >= pi:
        return x - 2 * pi
    return x


def add_hparams(summary, hparam_dict, metric_dict, hparam_domain_discrete=None, step=0):
    if type(hparam_dict) is not dict or type(metric_dict) is not dict:
        raise TypeError('hparam_dict and metric_dict should be dictionary.')
    exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

    summary.file_writer.add_summary(exp, global_step=step)
    summary.file_writer.add_summary(ssi, global_step=step)
    summary.file_writer.add_summary(sei, global_step=step)
    for k, v in metric_dict.items():
        summary.add_scalar(k, v, global_step=step)
