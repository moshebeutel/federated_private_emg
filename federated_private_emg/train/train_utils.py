from __future__ import annotations

from dataclasses import astuple

import torch
import wandb
from tqdm import tqdm

# from common.config import Config
from common.utils import labels_to_consecutive, calc_grad_norm
from differential_privacy.params import DpParams
from differential_privacy.utils import add_dp_noise
from train.params import TrainParams
from train.train_objects import TrainObjects


def run_single_epoch_return_grads(train_objects: TrainObjects,
                                  batch_size: int) -> (float, float):
    device = next(train_objects.model.parameters()).device

    assert train_objects.model.training, 'Model not in train mode'
    running_loss, correct_counter, sample_counter, counter = 0, 0, 0, 0

    y_pred, y_labels = [], []
    for k, batch in enumerate(train_objects.loader):
        curr_batch_size = batch[1].size(0)
        if curr_batch_size < batch_size:
            continue
        counter += 1

        sample_counter += curr_batch_size
        batch = (t.to(device) for t in batch)
        emg, labels = batch
        labels = labels_to_consecutive(labels).squeeze()

        outputs = train_objects.model(emg.float())

        loss = train_objects.criterion(outputs, labels.long())
        running_loss += float(loss)
        _, predicted = torch.max(outputs.data, 1)
        if train_objects.model.training:
            loss.backward()

        correct = (predicted == labels).sum().item()
        correct_counter += int(correct)

        y_pred += predicted.cpu().tolist()
        y_labels += labels.cpu().tolist()
    loss = running_loss / float(counter)
    acc = 100 * correct_counter / sample_counter
    return loss, acc


def run_single_epoch(train_objects: TrainObjects,
                     train_params: TrainParams,
                     dp_params: DpParams = None) -> (float, float):
    device = next(train_objects.model.parameters()).device
    _, batch_size, descent_every, _, _, add_dp_noise_before_optimization = astuple(train_params)
    assert not train_objects.model.training or train_objects.optimizer is not None, \
        'None Optimizer  given at train epoch'
    running_loss, correct_counter, sample_counter, counter = 0, 0, 0, 0

    y_pred, y_labels = [], []
    for k, batch in enumerate(train_objects.loader):
        curr_batch_size = batch[1].size(0)
        if curr_batch_size < batch_size:
            continue
        counter += 1

        sample_counter += curr_batch_size
        batch = (t.to(device) for t in batch)
        emg, labels = batch
        labels = labels_to_consecutive(labels).squeeze()

        if train_objects.model.training and \
                (descent_every == 1 or
                 (descent_every > 1 and ((k + 1) % descent_every) == 1)):
            # optimizer.zero_grad() done
            # at first batch or if optimizer.step() done at previous batch

            train_objects.optimizer.zero_grad()

        outputs = train_objects.model(emg.float())

        loss = train_objects.criterion(outputs, labels.long()) / descent_every
        running_loss += float(loss)
        _, predicted = torch.max(outputs.data, 1)
        if train_objects.model.training:
            loss.backward()
            if descent_every == 1 or descent_every > 1 and ((k + 1) % descent_every) == 0:
                if add_dp_noise_before_optimization:
                    add_dp_noise(train_objects.model, params=dp_params)
                train_objects.optimizer.step()
        correct = (predicted == labels).sum().item()
        correct_counter += int(correct)

        y_pred += predicted.cpu().tolist()
        y_labels += labels.cpu().tolist()
    loss = running_loss / float(counter)
    acc = 100 * correct_counter / sample_counter
    return loss, acc


def train_model(train_objects: TrainObjects,
                validation_loader,
                test_loader,
                train_params: TrainParams,
                dp_params=None,
                log2wandb: bool = True,
                show_pbar: bool = True,
                output_fn=lambda s: None):
    num_epochs, _, _, validation_eval_every, test_eval_at_end, add_dp_noise_before_optimization = astuple(train_params)
    eval_params = TrainParams(epochs=1, batch_size=train_params.batch_size)
    train_loss, train_acc, validation_loss, validation_acc = 0, 0, 0, 0
    eval_num = 0
    range_epochs = range(1, num_epochs + 1)
    epoch_pbar = tqdm(range_epochs) if show_pbar else None
    train_objects.model.train()
    for epoch in (epoch_pbar if show_pbar else range_epochs):

        epoch_train_loss, epoch_train_acc = \
            run_single_epoch(train_objects=train_objects,
                             train_params=train_params,
                             dp_params=dp_params)

        if validation_eval_every == 1 or (validation_eval_every > 1 and epoch % (validation_eval_every - 1)) == 0:
            train_objects.model.eval()
            epoch_validation_loss, epoch_validation_acc = \
                run_single_epoch(train_objects=TrainObjects(loader=validation_loader, model=train_objects.model),
                                 train_params=eval_params)
            train_objects.model.train()
            eval_num += 1
            validation_loss += epoch_validation_loss
            validation_acc += epoch_validation_acc
        if show_pbar:
            epoch_pbar.set_description(
                f'epoch {epoch} loss {round(epoch_train_loss, 3)} acc {epoch_train_acc}'
                f' epoch validation loss {round(epoch_validation_loss, 3)}'
                f' epoch validation acc {round(epoch_validation_acc, 3)}')

        train_acc += epoch_train_acc
        train_loss += epoch_train_loss

        if log2wandb:
            wandb.log({'epoch': epoch,
                       'epoch_train_loss': epoch_train_loss,
                       'epoch_train_acc': epoch_train_acc,
                       'epoch_validation_loss': epoch_validation_loss,
                       'epoch_validation_acc': epoch_validation_acc,
                       'train_loss': train_loss / epoch,
                       'train_acc': train_acc / epoch,
                       'validation_loss': validation_loss / eval_num,
                       'validation_acc': validation_acc / eval_num
                       })
    if test_eval_at_end:
        train_objects.model.eval()
        test_loss, test_acc = run_single_epoch(
            train_objects=TrainObjects(loader=test_loader, model=train_objects.model),
            train_params=eval_params)
        output_fn(f'Test Loss {test_loss} Test Acc {test_acc}')
        if log2wandb:
            wandb.log({'test_loss': test_loss, 'test_acc': test_loss})

    return train_loss / num_epochs, train_acc / num_epochs, validation_loss / eval_num, validation_acc / eval_num
