from typing import Callable, Optional

import torch
from tqdm import tqdm

from utils import ModelSettings, CheckpointManager, BuilderManager


def train(model, dataloader, criterion, optimizer, scheduler, checkpoint = None, dynamic_update = True,
          auto_apply = True, process_bar = None, log_func: Optional[Callable] = None):
    config = ModelSettings.load_config()
    device = ModelSettings.device
    if checkpoint is None:
        checkpoint = {'epoch': 0, 'min_loss': 0.0, 'max_accuracy': 0.0}
    min_loss, max_accuracy, start_epoch = checkpoint['min_loss'], checkpoint['max_accuracy'], checkpoint['epoch']
    checkpoint_manager = CheckpointManager()
    num_epochs = config['train']['num_epochs']
    save_interval = 1 if config['train']['save_interval'] is None else config['train']['save_interval']
    self_evaluate_interval = 5 if config['train']['self_evaluate_interval'] is None else config['train'][
        'self_evaluate_interval']
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.
        avg_loss = 0.

        if dynamic_update:
            ModelSettings.setup_training(auto_apply = auto_apply)

        train_process_bar = (
            process_bar if process_bar is not None else
            BuilderManager.get_process_bar(config, 'train', dataloader[0]) or
            tqdm(enumerate(dataloader[0]), total = len(dataloader[0]), desc = 'Training: ', colour = 'red')
        )

        for i, (images, labels) in train_process_bar:
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.unsqueeze(1).float().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (i + 1)

            train_process_bar.set_description(
                f"\033[38;5;22m\033[47mEpoch {epoch + 1}/{num_epochs}\033[0m, \033[92mBatch {i + 1}/{len(dataloader[0])}\033[0m, \033[96mLoss: {avg_loss:.6f}\033[0m")

        tqdm.write(f"\033[92mEpoch {epoch + 1}/{num_epochs}\033[0m, \033[96mAverage Loss: {avg_loss:.6f}\033[0m")

        scheduler.step()

        val_loss, val_accuracy = evaluate(model, dataloader[1], criterion)

        min_loss, max_accuracy = checkpoint_manager.save_weights(model, min_loss, max_accuracy, val_loss, val_accuracy)

        train_accuracy = None
        if (epoch + 1) % self_evaluate_interval == 0:
            self_evaluate_process_bar = BuilderManager.get_process_bar(config, 'self-evaluate',
                                                                       dataloader[0]) or tqdm(
                enumerate(dataloader[0]), total = len(dataloader[0]), desc = 'Self-evaluating', colour = 'yellow')

            _, train_accuracy = evaluate(model, dataloader[0], criterion, device,
                                         process_bar = self_evaluate_process_bar)

        checkpoint_manager.save_checkpoint(model, optimizer, epoch + 1, min_loss, max_accuracy, save_interval)

        if log_func is not None:
            log_func(epoch + 1, val_loss, val_accuracy, avg_loss, train_accuracy)

    print(
        f'\033[92mTraining complete. Best validation accuracy: \033[95m{max_accuracy}\033[92m, with minimum loss: \033[96m{min_loss}\033[0m.')


def evaluate(model, dataloader, criterion, dynamic_auc = False, process_bar = None):
    config = ModelSettings.load_config()
    device = ModelSettings.device
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    evaluate_process_bar = (
        process_bar if process_bar is not None else
        BuilderManager.get_process_bar(config, 'evaluate', dataloader) or
        tqdm(enumerate(dataloader), total = len(dataloader), desc = 'Evaluating: ', colour = 'blue')
    )

    with torch.no_grad():
        for i, (images, labels) in evaluate_process_bar:
            images = images.to(device)
            labels = labels.unsqueeze(1).float().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            predicted = (probs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if dynamic_auc:
                evaluate_process_bar.set_description(
                    f"\033[92mBatch {i + 1}/{len(dataloader)}\033[0m, \033[95mCurrent Accuracy: {100 * correct / total:.4f}%\033[0m, \033[96mLoss: {loss:.6f}\033[0m")

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    tqdm.write(f"\033[96mValidation Loss: {avg_loss:.6f}\033[0m, \033[95mAccuracy: {100 * accuracy:.4f}%\033[0m",
               file = None)

    return avg_loss, accuracy
