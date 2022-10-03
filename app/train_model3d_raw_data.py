from datetime import datetime
import logging
import os
import torch
from tqdm import tqdm
import wandb
from utils import config_logger, labels_to_consecutive
from federated_private_emg.fed_priv_models.model3d import Model3d

TENSORS_DATA_DIR = '../data/tensors_datasets'
WINDOWED_DATA_DIR = '../data/windowed_tensors_datasets'
NUM_CLASSES = 7
WINDOW_SIZE = 260

BATCH_SIZE = 16
NUM_WORKERS = 2
DEVICE = 'cpu'
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 200
WRITE_TO_WANDB = False
EVAL_EVERY = 1


@torch.no_grad()
def eval_model(loader, model, criterion):
    running_loss, correct_test, total_test, counter = 0, 0, 0, 0
    y_pred, y_labels = [], []
    for k, batch in enumerate(loader):
        curr_batch_size = batch[1].size(0)
        if curr_batch_size < BATCH_SIZE:
            continue
        counter += 1
        total_test += curr_batch_size
        batch = (t.to(DEVICE) for t in batch)
        emg, labels = batch
        labels = labels_to_consecutive(labels)

        outputs = model(emg.float())

        loss = criterion(outputs, labels.long())
        running_loss += float(loss)
        _, predicted = torch.max(outputs.data, 1)

        correct = (predicted == labels).sum().item()
        correct_test += int(correct)

        y_pred += predicted.cpu().tolist()

        y_labels += labels.cpu().tolist()
    test_loss = running_loss / float(counter)
    test_acc = 100 * correct_test / total_test
    return test_loss, test_acc


def train_model(criterion, model, optimizer, train_loader, test_loader):
    total_counter = 0
    epoch_pbar = tqdm(range(NUM_EPOCHS))
    model.train()
    for epoch in epoch_pbar:
        running_loss, correct_train, total_train, counter = 0, 0, 0, 0
        y_pred, y_labels = [], []
        for k, batch in enumerate(train_loader):
            curr_batch_size = batch[1].size(0)
            if curr_batch_size < BATCH_SIZE:
                continue
            counter += 1
            total_counter += curr_batch_size
            total_train += curr_batch_size
            batch = (t.to(DEVICE) for t in batch)
            emg, labels = batch
            labels = labels_to_consecutive(labels)

            optimizer.zero_grad()
            outputs = model(emg.float())

            loss = criterion(outputs, labels.long())
            running_loss += float(loss)
            _, predicted = torch.max(outputs.data, 1)
            loss.backward()
            optimizer.step()
            correct = (predicted == labels).sum().item()
            correct_train += int(correct)

            y_pred += predicted.cpu().tolist()
            y_labels += labels.cpu().tolist()
        train_loss = running_loss / float(counter)
        train_acc = 100 * correct_train / total_train
        epoch_pbar.set_description(f'epoch {epoch} loss {train_loss} acc {train_acc}')
        # if epoch % (EVAL_EVERY - 1) == 0:
        test_loss, test_acc = eval_model(test_loader, model, criterion)
        epoch_pbar.set_description(
            f'epoch {epoch} loss {train_loss} acc {train_acc} test loss {test_loss} test acc {test_acc}')

        if WRITE_TO_WANDB:
            wandb.log({'epoch': epoch,
                       'train_loss': train_loss,
                       'train_acc': train_acc,
                       'test_loss': test_acc,
                       'test_acc': test_acc
                       })


def main():
    n = datetime.now()
    time_str = f'_{n.year}_{n.month}_{n.day}_{n.hour}_{n.minute}_{n.second}'
    exp_name = os.path.basename(__file__)[:-3] + time_str
    logger = config_logger(f'{exp_name}_logger',
                           level=logging.INFO, log_folder='../log/')

    if WRITE_TO_WANDB:
        wandb.init(project="emg_gp_moshe", entity="emg_diff_priv", name=exp_name)
        # wandb.config.update({})

    logger.info(exp_name)

    train_x = torch.load(os.path.join(WINDOWED_DATA_DIR, 'X_train_windowed.pt'))
    assert train_x.shape[1] == WINDOW_SIZE, f'Expected windowed data with window size {WINDOW_SIZE}.' \
                                            f' Got {train_x.shape[1]}'
    assert train_x.shape[0] >= BATCH_SIZE, f'Batch size is greater than dataset. ' \
                                           f'Batch size {BATCH_SIZE}, Dataset {train_x.shape[0]}'
    train_y = torch.load(os.path.join(WINDOWED_DATA_DIR, 'y_train_windowed.pt'))
    assert train_x.shape[0] == train_y.shape[0], f'Found {train_y.shape[0]} labels for dataset size {train_x.shape[0]}'
    assert train_y.dim() == 1 or train_y.shape[1] == 1, f'Labels expected to have one dimension'

    logger.info(f'Loaded train_x shape {train_x.shape} train_y shape {train_y.shape}')

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    logger.info(f'Train Loader created, Len {len(train_loader)}')

    test_x = torch.load(os.path.join(WINDOWED_DATA_DIR, 'X_test_windowed.pt'))
    test_y = torch.load(os.path.join(WINDOWED_DATA_DIR, 'y_test_windowed.pt'))

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    model = Model3d(number_of_classes=NUM_CLASSES, window_size=WINDOW_SIZE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    train_model(criterion, model, optimizer, train_loader, test_loader)


if __name__ == '__main__':
    main()
