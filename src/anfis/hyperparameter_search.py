import os
from functools import partial

import numpy as np
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import FunctionStopper

from anfis.consequent_layer import ConsequentLayerType
from anfis.utils import calc_error, save_fuzzy_membership_functions


def train_anfis(config: dict, train_data=None, criterion=None, model_creator=None, num_epochs=1000,
                checkpoint_dir=None, summaries_dict=None):
    config.setdefault('h', ConsequentLayerType.PLAIN)
    config.setdefault('n', False)
    config.setdefault('m', 0.99)
    config.setdefault('wd', 0)
    config.setdefault('d', 0)

    if callable(train_data):
        train_data = train_data(config)

    net = model_creator(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
    net.to(device)

    # name = []
    # for (k,v) in config.items():
    #     name.append(f"{k}{v}")
    #
    # checkpoint_dir = os.path.join(checkpoint_dir, "-".join(name))

    optimizer = make_optimizer(net, config['lr'], config['m'], config['d'], config['wd'],
                               config['n'])

    if checkpoint_dir:
        if config['h']:
            model_state, optimizer_state, consequent_coeffs = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
            net.load_state_dict(model_state)
            net.layer['consequent'].coeff = consequent_coeffs

            optimizer.load_state_dict(optimizer_state)
        else:
            model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
            net.load_state_dict(model_state)

            optimizer.load_state_dict(optimizer_state)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0

        for i, data in enumerate(train_data, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1

        # print("[%d] loss: %.3f" % (epoch + 1, running_loss / epoch_steps))

        # Epoch ending, so now fit the coefficients based on all data:
        x, y_actual = train_data.dataset.tensors
        with torch.no_grad():
            net.fit_coeff(x, y_actual)

        y_pred = net(x)
        loss = criterion(y_pred, y_actual)
        mse, rmse, perc_loss, rsq = calc_error(y_pred, y_actual)

        if not torch.isfinite(loss):

            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")

                save_fuzzy_membership_functions(net, os.path.join(checkpoint_dir, 'mfs.txt'))

                if config['h']:
                    torch.save((net.state_dict(), optimizer.state_dict(), net.layer['consequent'].coeff), path)
                else:
                    torch.save((net.state_dict(), optimizer.state_dict()), path)

            # summary_name = str(config).replace(" ", "").replace(":", "=").replace("{", "").replace("}", "")
            #
            # # print(summaries_dict, len(summaries_dict))
            #
            # if summary_name in summaries_dict:
            #     summary = summaries_dict[summary_name]
            # else:
            #     summary = SummaryWriter(os.path.join(str(Path(checkpoint_dir).parent.absolute()), "hparams"))
            #     summaries_dict[summary_name] = summary
            #
            # summary.add_hparams({
            #     "Learning Rate": config['lr'],
            #     "Hybrid": config['h'],
            #     "Momentum": config['m'],
            #     "Dampening": config['d'],
            #     "Weight Decay": config['wd'],
            #     "Nesterov": config['n']
            # }, {
            #     "Loss": loss, "RMSE": rmse, "Percent Loss": perc_loss
            # }, run_name="ANFIS")
            # summary.flush()

        # print(type(loss.cpu().item()))

        # if torch.isnan(loss):
        #     loss = torch.FloatTensor([1e9])
        #     rmse = 1e9
        #     perc_loss = torch.FloatTensor([1e9])

        tune.report(loss=loss.cpu().item(), mse=mse.cpu().item(), rmse=rmse, perc_loss=perc_loss.cpu().item(), rsq=rsq)

    print("Finished training")


def main(data, config, model_creator, criterion, folder_name, num_samples=10, max_num_epochs=1000, num_workers=4,
         gpus_per_trial=0, resume=True, grace_period=200, num_checkpoints=50, output_dir='./runs/'):
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=grace_period,
        reduction_factor=2)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "rmse", "Percent Loss", "training_iteration"])

    if not os.path.exists(os.path.join(output_dir, folder_name)):
        resume = False
        print("Previous search instance does not exist so recreating Tuner")
    else:
        print("Previous checkpoints exist")

    summaries_dict = dict()

    stopper = FunctionStopper(lambda id, results: not np.isfinite(results['loss']))

    result = tune.run(
        partial(train_anfis, train_data=data, model_creator=model_creator, criterion=criterion,
                num_epochs=max_num_epochs, summaries_dict=summaries_dict),
        resources_per_trial={"cpu": num_workers, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=output_dir,
        name=folder_name,
        resume=resume,
        max_failures=10,
        stop=stopper,
        keep_checkpoints_num=num_checkpoints,
        checkpoint_freq=1,
        verbose=1,
        raise_on_failed_trial=False,
        checkpoint_score_attr='min-loss')

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))


def make_optimizer(model, lr=1e-4, momentum=0.99, dampening=0, weight_decay=0, nesterov=False):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening,
                                weight_decay=weight_decay, nesterov=nesterov)

    return optimizer
