import glob
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from anfis.anfis import JointAnfisNet
from anfis.consequent_layer import ConsequentLayerType
from anfis.utils import plot_fuzzy_variables, plot_fuzzy_consequent, plot_fuzzy_membership_functions, calc_error, \
    plot_summary_results, save_fuzzy_membership_functions
from vizualize.auto_grad_viz import get_gradient_values
from vizualize.auto_grad_viz import make_dot


def profile_model(log_dir, model, data, optimizer, criterion, wait=1, warm_up=1, active=5, repeat=2):
    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warm_up, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=True,
            with_stack=True, profile_memory=False) as prof:

        # Process each mini-batch in turn:
        for step, (x, y_actual) in enumerate(data):
            print(step)
            if step >= (wait + warm_up + active) * repeat:

                dtaas = glob.glob(f"{log_dir}/*.json")
                for dtaa in dtaas:
                    lines = []

                    with open(dtaa, 'r') as a:
                        for a in a.readlines():
                            lines.append(a.replace('\\', '\\\\'))
                    with open(dtaa, 'w') as a:
                        a.writelines(lines)

                break

            y_pred = model(x)
            # Compute and print loss
            loss = criterion(y_pred, y_actual)
            # Zero gradients, perform a backward pass, and update the weights.

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            prof.step()

            step += 1


def train_anfis_with(model, data, optimizer, criterion,
                     epochs=500, show_plots=False, summary: SummaryWriter = False, save_on_lowest_loss=True,
                     summary_frequency=10, summary_figure_frequency=50, checkpoint_save_frequency=100,
                     show_backpropagation=False, show_backpropagation_values=False):
    """
        Train the given model using the given (x,y) data.
    """
    print('### Training for {} epochs, training size = {} cases'.
          format(epochs, data.dataset.tensors[0].shape[0]))

    layer_type = model.rules_type

    if save_on_lowest_loss:
        checkpoint = os.path.join(summary.get_logdir(), 'checkpoint.pth')
        min_loss = None
        save_epoch = -1
        new_model = None

    if summary is not None:
        if layer_type == ConsequentLayerType.PLAIN or layer_type == ConsequentLayerType.MAMDANI:
            print("Creating graph")
            summary.add_graph(model, data.dataset.tensors[0])
            summary.flush()

        plot_fuzzy_consequent(summary, model, -1)
        plot_fuzzy_membership_functions(summary, model, -1)

    if show_backpropagation:
        parameters = None

    for t in range(epochs):
        # Process each mini-batch in turn:
        for x, y_actual in data:
            y_pred = model(x)
            # Compute and print loss
            loss = criterion(y_pred, y_actual)
            # Zero gradients, perform a backward pass, and update the weights.

            optimizer.zero_grad()

            if show_backpropagation:
                if show_backpropagation_values:
                    parameters = get_gradient_values(loss)
                    show_backpropagation_values = False

            loss.backward()

            if show_backpropagation:
                g = make_dot(loss, dict(model.named_parameters()), parameters)
                g.view()
                time.sleep(5)

            optimizer.step()
        # Epoch ending, so now fit the coefficients based on all data:
        x, y_actual = data.dataset.tensors
        with torch.no_grad():
            if model.rules_type == ConsequentLayerType.HYBRID:
                model.fit_coeff(x, y_actual)
            elif model.rules_type == ConsequentLayerType.SYMMETRIC and t >= 50:
                model.fit_coeff(t)
            elif model.rules_type == ConsequentLayerType.PLAIN:
                model.fit_coeff(x, y_actual)
        # Get the error rate for the whole batch:
        y_pred = model(x)
        mse, rmse, perc_loss, rsq = calc_error(y_pred, y_actual)
        # Print some progress information as the net is trained:

        if save_on_lowest_loss:
            if (min_loss is None or min_loss > mse):
                min_loss = mse
                save_epoch = t
                new_model = model.state_dict()

            if t % checkpoint_save_frequency == 0 and new_model is not None:
                new_model = None
                print("Saved model checkpoint. Loss:", min_loss.item())

                save_fuzzy_membership_functions(model, os.path.join(summary.get_logdir(), 'mfs.txt'))

                save_anfis(model, checkpoint)

        if epochs < 30 or t % summary_frequency == 0:
            print('epoch {:4d}: MSE={:.5f}, RSQ={:.5f}, RMSE={:.5f} ={:.2f}%'
                  .format(t, mse, rsq, rmse, perc_loss))

            if summary is not None:
                loss = criterion(y_pred, y_actual)
                summary.add_scalar('Loss', loss, t)
                summary.add_scalar('Errors/MSE', mse, t)
                summary.add_scalar('Errors/RMSE', rmse, t)
                summary.add_scalar('Errors/Percent Loss', perc_loss, t)
                summary.add_scalar('Errors/RSQ', rsq, t)

                plot_fuzzy_variables(summary, model, t)

        if summary is not None and t % summary_figure_frequency == 0:
            plot_fuzzy_consequent(summary, model, t)
            plot_fuzzy_membership_functions(summary, model, t)

    if save_on_lowest_loss:
        save_anfis(model, checkpoint)
        print("Saved model checkpoint:")
        print("Loss:", min_loss.item())
        print("Epoch:", save_epoch)
        print("Location:", checkpoint)

    # End of training, so graph the results:
    if summary is not None:
        plot_fuzzy_variables(summary, model, epochs - 1)
        plot_fuzzy_consequent(summary, model, epochs - 1)
        plot_fuzzy_membership_functions(summary, model, epochs - 1)

        y_actual = data.dataset.tensors[1]
        y_pred = model(data.dataset.tensors[0])

        plot_summary_results(summary, y_pred, y_actual)

        summary.close()


def train_anfis(model, data, epochs=500, show_plots=False, lr=1e-4, momentum=0.99, dampening=0, weight_decay=0,
                nesterov=False, summary=None, show_backpropagation=False, show_backpropagation_values=False,
                summary_frequency=10, summary_figure_frequency=50, checkpoint_save_frequency=100):
    """
        Train the given model using the given (x,y) data.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening,
                                weight_decay=weight_decay, nesterov=nesterov)
    criterion = torch.nn.MSELoss(reduction='sum')
    train_anfis_with(model, data, optimizer, criterion, epochs, show_plots, summary,
                     show_backpropagation=show_backpropagation, show_backpropagation_values=show_backpropagation_values,
                     summary_frequency=summary_frequency, summary_figure_frequency=summary_figure_frequency,
                     checkpoint_save_frequency=checkpoint_save_frequency)


def save_anfis(model, checkpoint):
    layer_type = model.rules_type

    if layer_type == ConsequentLayerType.HYBRID:
        torch.save({"model_state_dict": model.state_dict(),
                    "consequent_coeffs": model.layer['consequent'].coeff}, checkpoint)
    else:
        torch.save(model.state_dict(), checkpoint)


def load_anfis(model, file):
    uses_hybrid = model.rules_type == ConsequentLayerType.HYBRID

    data_dict = torch.load(file)

    if uses_hybrid:
        model.load_state_dict(data_dict['model_state_dict'])
        model.layer['consequent'].coeff = data_dict['consequent_coeffs']
    else:
        model.load_state_dict(data_dict)


def make_joint_anfis(variable_joint_fuzzy_definitons, outputs, rules_type=ConsequentLayerType.PLAIN, mamdani_defs=None):
    model = JointAnfisNet('Simple joint classifier', variable_joint_fuzzy_definitons, outputs, rules_type=rules_type,
                          mamdani_defs=mamdani_defs)

    return model
