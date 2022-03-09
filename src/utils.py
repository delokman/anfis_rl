from typing import List, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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


class DoNothing:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def reward_function_grid_visualization(variable_ranges: List[np.ndarray], variable_names: List[str],
                                       reward_function: Callable[..., float], res: int = 50) -> Tuple[Figure, Figure]:
    """
    Implemented function to visualize the reward output of each of the variables, related with each other as contour plots

    Example:
    
    >>> def reward(a, b, c, d):
    ...     return a + b + c
    >>> variable_ranges = [
    ...    np.linspace(0, 1),
    ...    np.linspace(1, 2),
    ...    np.linspace(2, 3),
    ...    np.linspace(0, 1),
    ... ]
    >>> variable_names = ["A", "B", "C", "D"]
    >>> fig = reward_function_grid_visualization(variable_ranges, variable_names, reward)
    >>> plt.show()

    :param variable_ranges: the variables ranges to calculate the rewards for
    :param variable_names: the names of the variables to place in the axes of the figure
    :param reward_function: the reward function to use to plot the output
    :param res: the resolution of the contour lines of the figure
    :return: The reward grid figure
    """

    n_plots = len(variable_ranges)
    fig, axs = plt.subplots(nrows=n_plots, ncols=n_plots)
    grads_fig, grads_axs = plt.subplots(nrows=n_plots, ncols=n_plots)

    empty = [0 for _ in range(n_plots)]

    mins = []
    maxes = []
    contours = []
    mins_g = []
    maxes_g = []
    contours_g = []

    for y in range(n_plots):
        for x in range(n_plots):
            x_d = variable_ranges[x]
            y_d = variable_ranges[y]

            xx, yy = np.meshgrid(x_d, y_d, indexing='ij')

            data = np.zeros((x_d.shape[0], y_d.shape[0]))
            for i in range(x_d.shape[0]):
                for j in range(y_d.shape[0]):
                    d = empty[:]

                    if x != y:
                        d[x] = x_d[i]
                        d[y] = y_d[j]
                        data[i, j] = reward_function(*d)
                    else:
                        d[x] = x_d[i]
                        data[i, j] = reward_function(*d)
                        d[y] = y_d[j]
                        data[i, j] += reward_function(*d)

            ax: Axes = axs[y, x]

            min_z = np.min(data)
            max_z = np.max(data)

            mins.append(min_z)
            maxes.append(max_z)

            c = ax.contourf(xx, yy, data, res, cmap='seismic')
            contours.append(c)

            # ax.contour(xx, yy, data, res // 5)

            dx = (x_d[-1] - x_d[0]) / x_d.shape[0]
            dy = (y_d[-1] - y_d[0]) / y_d.shape[0]

            grad_reward_x, grad_reward_y = np.gradient(data, dx, dy)
            grad_ax = grads_axs[y, x]

            z = grad_reward_x + grad_reward_y

            min_z = np.min(z)
            max_z = np.max(z)

            maxes_g.append(max_z)
            mins_g.append(min_z)

            c = grad_ax.contourf(xx, yy, z, res, cmap='coolwarm')
            contours_g.append(c)

            n = 5
            skip = (slice(None, None, n), slice(None, None, n))

            grad_ax.quiver(xx[skip], yy[skip], grad_reward_x[skip], grad_reward_y[skip], scale=None, scale_units='xy',
                           angles='xy')

            grad_ax: Axes
            # grad_ax.pcolor(xx,yy, z ,vmin=-1, vmax=1)

            bot = False
            side = False
            if y == n_plots - 1:
                bot = True
                ax.set_xlabel(variable_names[x])
                grad_ax.set_xlabel(variable_names[x])
            if x == 0:
                side = True
                ax.set_ylabel(variable_names[y])
                grad_ax.set_ylabel(variable_names[y])

            ax.tick_params(labelbottom=bot, labelleft=side)
            fig: Figure
            grad_ax.tick_params(labelbottom=bot, labelleft=side)

    min_z = min(*mins, -0.01)
    max_z = max(*maxes, 0.01)
    divnorm = colors.TwoSlopeNorm(vmin=min_z, vcenter=0., vmax=max_z)

    for c in contours:
        c.set_norm(divnorm)

    min_z = min(*mins_g, -0.01)
    max_z = max(*maxes_g, 0.01)
    divnorm_grads = colors.TwoSlopeNorm(vmin=min_z, vcenter=0., vmax=max_z)

    for c in contours_g:
        c.set_norm(divnorm_grads)

    # grads_fig.subplots_adjust(.04, .052, .995, .983, .043, .057)
    # fig.subplots_adjust(.04, .052, .995, .983, .043, .057)

    # fig.tight_layout()
    return fig, grads_fig
