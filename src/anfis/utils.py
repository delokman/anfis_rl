# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS in torch: some simple functions to supply data and plot results.
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from anfis.consequent_layer import ConsequentLayerType
from anfis.joint_membership import JointMembership

dtype = torch.float


def plotErrors(errors):
    """
        Plot the given list of error rates against no. of epochs
    """
    plt.plot(range(len(errors)), errors, '-ro', label='errors')
    plt.ylabel('Percentage error')
    plt.xlabel('Epoch')
    plt.show()


def plotResults(y_actual, y_predicted):
    """
        Plot the actual and predicted y values (in different colours).
    """
    plt.plot(range(len(y_predicted)), y_predicted.detach().numpy(),
             'r', label='trained')
    plt.plot(range(len(y_actual)), y_actual.numpy(), 'b', label='original')
    plt.legend(loc='upper left')
    plt.show()


def _plot_mfs(var_name, fv, x):
    """
        A simple utility function to plot the MFs for a variable.
        Supply the variable name, MFs and a set of x values to plot.
    """
    # Sort x so we only plot each x-value once:
    xsort, _ = x.sort()
    for mfname, yvals in fv.fuzzify(xsort):
        plt.plot(xsort.tolist(), yvals.tolist(), label=mfname)
    plt.xlabel('Values for variable {} ({} MFs)'.format(var_name, fv.num_mfs))
    plt.ylabel('Membership')
    plt.legend(bbox_to_anchor=(1., 0.95))
    plt.show()


def plot_all_mfs(model, x):
    for i, (var_name, fv) in enumerate(model.layer.fuzzify.varmfs.items()):
        _plot_mfs(var_name, fv, x[:, i])


def calc_error(y_pred, y_actual):
    with torch.no_grad():
        tot_loss = F.mse_loss(y_pred, y_actual)
        rmse = torch.sqrt(tot_loss).item()
        perc_loss = torch.mean(100. * torch.abs((y_pred - y_actual)
                                                / y_actual))

        ss_total = torch.sum((y_actual - torch.mean(y_actual)) ** 2)
        ss_regression = torch.sum((y_actual - y_pred) ** 2)
        rsq = (1 - (ss_regression / ss_total)) * 100

    return (tot_loss, rmse, perc_loss, rsq)


def test_anfis(model, data, show_plots=False):
    """
        Do a single forward pass with x and compare with y_actual.
    """
    x, y_actual = data.dataset.tensors
    if show_plots:
        plot_all_mfs(model, x)
    print('### Testing for {} cases'.format(x.shape[0]))
    y_pred = model(x)
    mse, rmse, perc_loss, rsq = calc_error(y_pred, y_actual)
    print('MS error={:.5f}, RSQ error={:.5f}, RMS error={:.5f}, percentage={:.2f}%'
          .format(mse, rsq, rmse, perc_loss))
    if show_plots:
        plotResults(y_actual, y_pred)


def plot_fuzzy_variables(summary, model, epoch):
    for i, (var_name, fv) in enumerate(model.layer.fuzzify.varmfs.items()):
        for p in fv.named_parameters():
            summary.add_scalar(f"{var_name}/{p[0]}", p[1], epoch)

    if model.rules_type == ConsequentLayerType.MAMDANI:
        for name, value in model.layer['consequent'].mamdani_defs.named_parameters():
            summary.add_scalar(f"Consequent/{name}", value, epoch)


def plot_fuzzy_consequent(summary, model, t):
    with torch.no_grad():
        if model.rules_type == ConsequentLayerType.MAMDANI:
            consque = model.layer['consequent'].mamdani_defs
            consque.cache()

            values = consque.cache_output_values

            fig, ax = plt.subplots()

            s = 1

            for key, value in values.items():
                ax.plot([value - 1 / s, value, value + 1 / s], [0, 1, 0], label=consque.names[key])

            ax.legend()

            summary.add_figure("Consequent/Mamdani", fig, t)

        else:
            coeff = model.layer['consequent'].coeff
            coeff, bias = coeff[:, :, :-1], coeff[:, :, -1]
            names = ['Coeffs', "Bias"]

            for name, data in zip(names, [coeff, bias]):
                height, width = data.size()[0], data.size()[-1]

                offset = 0

                x_v = np.arange(max(width + offset, 2))
                y_v = np.arange(height + offset)
                X, Y = np.meshgrid(x_v, y_v)

                Z = data.detach().numpy().reshape((height, width))[::-1]

                fig, ax = plt.subplots()

                c = ax.pcolormesh(X, Y, Z, shading='auto', edgecolors='k', linewidth=0.1)
                ax.set_yticks(y_v)
                fig.colorbar(c, ax=ax)

                summary.add_figure(f"Consequent/{name}", fig, t)


def plot_fuzzy_membership_functions(summary, model, t, n=1000):
    with torch.no_grad():
        for i, (var_name, fv) in enumerate(model.layer.fuzzify.varmfs.items()):
            fv: JointMembership

            membership_range = fv.right_x() - fv.left_x()
            offset = membership_range * .2

            left = fv.left_x() - offset
            right = fv.right_x() + offset

            x = torch.linspace(left, right, steps=n, dtype=model.dtype).unsqueeze(1)

            fig, ax = plt.subplots()

            for mfname, yvals in fv.fuzzify(x):
                ax.plot(x.tolist(), yvals.tolist(), label=mfname)

            ax.legend()

            summary.add_figure(f"Membership/{var_name}", fig, t)


def save_fuzzy_membership_functions(model, file_name='mfs.txt'):
    with open(file_name, 'w') as mfs_file:
        for i, (var_name, fv) in enumerate(model.layer.fuzzify.varmfs.items()):

            parameters = fv.named_parameters()

            mfs_file.write(var_name + '\n')

            for p in parameters:
                mfs_file.write(f'{p[0]}={p[1].item()}\n')

        if model.rules_type == ConsequentLayerType.MAMDANI:
            mfs_file.write('mamdani\n')
            for name, value in model.layer['consequent'].mamdani_defs.named_parameters():
                mfs_file.write(f'{name}={value.item()}\n')


def plot_summary_results(summary: SummaryWriter, y_predicted, y_actual):
    fig, ax = plt.subplots()

    ax.plot(range(len(y_predicted)), y_predicted.detach().numpy(),
            'r', label='trained')
    ax.plot(range(len(y_actual)), y_actual.numpy(), 'b', label='original')
    ax.legend(loc='upper left')

    summary.add_figure("Result/Results", fig)


def plot_critic_weights(summary, model, epoch):
    critic = model.critic

    for name, layer in critic.named_parameters():
        name = name.replace(".", "/")
        summary.add_histogram(name, layer, global_step=epoch)
