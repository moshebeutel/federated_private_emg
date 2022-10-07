from __future__ import annotations

import logging

import torch
import wandb
from tqdm import tqdm

from common.config import Config
from common.utils import labels_to_consecutive


def run_single_epoch(loader: torch.utils.data.DataLoader,
                     model: torch.nn.Module,
                     criterion: torch.nn.CrossEntropyLoss,
                     optimizer: torch.optim.Optimizer = None,
                     add_dp_noise_before_optimization: bool = False,
                     optimizer_step_every: int = 1) -> (float, float):
    assert not model.training or optimizer is not None, 'None Optimizer  given at train epoch'
    running_loss, correct_counter, sample_counter, counter = 0, 0, 0, 0

    y_pred, y_labels = [], []
    for k, batch in enumerate(loader):
        curr_batch_size = batch[1].size(0)
        if curr_batch_size < Config.BATCH_SIZE:
            continue
        counter += 1

        sample_counter += curr_batch_size
        batch = (t.to(Config.DEVICE) for t in batch)
        emg, labels = batch
        labels = labels_to_consecutive(labels).squeeze()

        if model.training and \
                (optimizer_step_every == 1 or
                 (optimizer_step_every > 1 and ((k+1) % optimizer_step_every) == 1)):
            # opimizer.zero_grad() done
            # at first batch or if optimizer.step() done at previous batch
            optimizer.zero_grad()
        outputs = model(emg.float())

        loss = criterion(outputs, labels.long()) / optimizer_step_every
        running_loss += float(loss)
        _, predicted = torch.max(outputs.data, 1)
        if model.training:
            loss.backward()
            if optimizer_step_every == 1 or optimizer_step_every > 1 and ((k+1) % optimizer_step_every) == 0:
                if add_dp_noise_before_optimization:
                    add_dp_noise(model)
                optimizer.step()
        correct = (predicted == labels).sum().item()
        correct_counter += int(correct)

        y_pred += predicted.cpu().tolist()
        y_labels += labels.cpu().tolist()
    loss = running_loss / float(counter)
    acc = 100 * correct_counter / sample_counter
    return loss, acc


def train_model(criterion, model, optimizer,
                train_loader,
                validation_loader,
                test_loader,
                num_epochs: int = 1,
                validation_eval_every: int = 1,
                test_eval_at_end: bool = True,
                add_dp_noise_before_optimization=False,
                log2wandb=True,
                show_pbar=True):
    train_loss, train_acc, validation_loss, validation_acc = 0, 0, 0, 0
    eval_num = 0
    range_epochs = range(1, num_epochs + 1)
    epoch_pbar = tqdm(range_epochs) if show_pbar else None
    model.train()
    for epoch in (epoch_pbar if show_pbar else range_epochs):

        epoch_train_loss, epoch_train_acc = \
            run_single_epoch(loader=train_loader, model=model, criterion=criterion,
                             optimizer=optimizer,
                             add_dp_noise_before_optimization=add_dp_noise_before_optimization,
                             optimizer_step_every=2)

        if validation_eval_every == 1 or (validation_eval_every > 1 and epoch % (validation_eval_every - 1)) == 0:
            model.eval()
            epoch_validation_loss, epoch_validation_acc = run_single_epoch(loader=validation_loader, model=model,
                                                                           criterion=criterion)
            model.train()
            eval_num += 1
            validation_loss += epoch_validation_loss
            validation_acc += epoch_validation_acc
        if show_pbar:
            epoch_pbar.set_description(
                f'epoch {epoch} loss {round(epoch_train_loss, 3)} acc {epoch_train_acc}'
                f' epoch validation loss {round(epoch_validation_loss, 3)}'
                f' epoch validation acc {round(epoch_validation_acc, 3)}')

        train_acc += epoch_train_acc
        train_loss += epoch_train_acc

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
        model.eval()
        test_loss, test_acc = run_single_epoch(loader=test_loader, model=model, criterion=criterion)
        if log2wandb:
            wandb.log({'test_loss': test_loss, 'test_acc': test_loss})

    return train_loss / num_epochs, train_acc / num_epochs, validation_loss / eval_num, validation_acc / eval_num


def add_dp_noise(model):
    grad_norm = 0
    for p in model.parameters():
        # Sum grid norms
        grad_norm += float(torch.linalg.vector_norm(p.grad))
    for p in model.parameters():
        # Clip gradients
        p.grad /= max(1, grad_norm / Config.DP_C)
        # Add DP noise to gradients
        noise = torch.randn_like(p.grad) * Config.DP_SIGMA * Config.DP_C
        p.grad += noise
        p.grad /= Config.BATCH_SIZE  # Averaging.Use batch as the 'Lot'
