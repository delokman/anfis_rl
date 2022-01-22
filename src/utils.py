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


def markdown_rule_table(anfis):
    vardefs = anfis.layer['fuzzify'].varmfs
    vardefs_names = list(vardefs.keys())

    rules = anfis.layer['rules'].mamdani_ruleset

    var_index = rules['variable_rule_index']
    mem_index = rules['membership_indices']
    out_index = rules['outputs_membership']

    out_name = anfis.layer['consequent'].mamdani_defs.names

    uses_vel = anfis.velocity

    num_variable = [i + 1 for i in range(len(var_index[0]))]

    title = "| Rule ID | " + " | ".join([f"Membership {i}" for i in num_variable]) + " | Turning Output |"
    separator = "| -------- | " + "|".join([f"------" for _ in num_variable]) + " | ------ |"

    if uses_vel:
        out_name_vel = anfis.layer['consequent'].mamdani_defs_vel.names
        out_index_vel = rules['outputs_membership_velocity']

        title += " Velocity Output |"
        separator += '------ |'
    else:
        out_name_vel = None
        out_index_vel = None

    rules = [title,
             separator]

    for i in range(len(var_index)):
        temp = []

        for var, mem in zip(var_index[i], mem_index[i]):
            name = vardefs_names[var]

            temp.append(f"{name} is {list(vardefs[name].mfdefs.keys())[mem]}")

        if uses_vel:
            out = f"{out_name[out_index[i][0]]} | {out_name_vel[out_index_vel[i][0]]}"
        else:
            out = out_name[out_index[i][0]]
        rules.append(f'| {i} | {" | ".join(temp)}|  {out} |')

    return '\n'.join(rules)
