import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchinfo import summary
from tqdm import tqdm

from augment import ImageDataset
from config import ModelSettings, Mapping, Interface
from optimizer import WarmupSchedulerWrapper


class CheckpointManager:
    def __init__(self):
        """
        Initialize the CheckpointManager with a directory for saving/loading checkpoints and weights.
        """
        self.config = ModelSettings.load_config()
        self.config_stem = str(Path(ModelSettings.config_name).stem)
        self.weights_dir = Path(self.config['path']['weights_dir'])
        self.checkpoint_dir = Path(self.config['path']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = ModelSettings.device

    def load_weights(self, model, unmatch=False, verbose=True):
        """ Load weights into the model from a specified path. """
        load_path = self.weights_dir
        load_path_full = load_path / f'{self.config_stem}_weights.pth'
        if load_path_full.is_file():
            if not unmatch:
                print(f'\n\033[94mLoading weights from {load_path_full}\033[0m') if verbose else None
                model.load_state_dict(torch.load(load_path_full, weights_only=False, map_location=self.device),
                                      strict=False)
            else:
                self.weights_unmatch_load(model, load_path_full)
        else:
            print(f'\n\033[94mNo weights found at {load_path_full}\033[0m')

    def save_weights(self, model, min_loss, max_accuracy, loss, accuracy):
        """ Save the model weights if the accuracy has improved. """
        max_accuracy_old = round(max_accuracy, 6)
        min_loss = min(min_loss, loss)
        save_dir = self.weights_dir
        save_path_full = save_dir / f'{self.config_stem}_weights.pth'
        save_dir.mkdir(parents=True, exist_ok=True)
        if round(accuracy, 6) > max_accuracy:
            max_accuracy = round(accuracy, 6)
            print(
                '\033[41m \033[42m \033[43m \033[44m \033[45m \033[46m \033[47m'
                f'\033[38;5;22mAccuracy improved from {100 * max_accuracy_old:.4f}% to {100 * max_accuracy:.4f}%, saving weights to {save_path_full}'
                '\033[46m \033[45m \033[44m \033[43m \033[42m \033[41m \033[0m')
            torch.save(model.state_dict(), save_path_full)
        elif accuracy == max_accuracy:
            print(f'Accuracy did not improve from {100 * max_accuracy:.4f}%')
        else:
            print(f'Accuracy did not improve from {100 * max_accuracy:.4f}%')
        return min_loss, max_accuracy

    def save_checkpoint(self, model, optimizer, epoch, min_loss, max_accuracy, save_interval=None):
        """
        Save a checkpoint of the model and optimizer state.
        """

        save_dir = self.checkpoint_dir
        save_path_full = save_dir / f'{self.config_stem}_checkpoint.pth'
        save_dir.mkdir(parents=True, exist_ok=True)

        state = {
            'epoch': epoch,
            'min_loss': min_loss,
            'max_accuracy': max_accuracy,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        immediate_save = save_interval is None or save_interval == 0

        if immediate_save or epoch % save_interval == 0:
            torch.save(state, save_path_full)
            if save_interval >= 5 or immediate_save:
                print(f'\033[94mCheckpoint saved at epoch {epoch} to {save_path_full}\033[0m')

        if save_interval < 5 and epoch % 5 == 0 and not immediate_save:
            print(
                f'\033[94mCheckpoint saving is functioning every {save_interval} epoch(s) at {save_path_full}.\033[0m')

    def load_checkpoint(self, model, optimizer=None, revised_value=None, use_saved_weights=False,
                        strict_load=True, verbose=True):
        """
        Load the model and optimizer state from a checkpoint file.
        """
        if revised_value is None:
            revised_value = []
        load_path = self.checkpoint_dir
        load_path_full = load_path / f'{self.config_stem}_checkpoint.pth'
        epoch = self.config['train']['num_epochs']
        if load_path_full.is_file():
            print(f'\033[94m\nLoading checkpoints from {load_path_full}\033[0m') if verbose else None
            checkpoint = torch.load(load_path_full, weights_only=True, map_location=self.device)
            last_epoch = checkpoint['epoch']
            if revised_value is not None:
                checkpoint['min_loss'], checkpoint['max_accuracy'] = revised_value[0], revised_value[1]
                print(
                    f'\033[94mLast epoch: {last_epoch}, revised_accuracy: {100 * checkpoint["max_accuracy"]:.4f}%\n\033[0m') if verbose else None
            else:
                print(
                    f'\033[94mLast epoch: {last_epoch}, max_accuracy: {100 * checkpoint["max_accuracy"]:.4f}%\n\033[0m') if verbose else None

            if epoch is not None and last_epoch >= epoch:
                print(
                    f'\033[94mTraining is already completed at epoch {last_epoch}. You may increase num_epochs to continue training.\033[0m')
                response = input('\033[94mWould you like to start a new training session? 【y/n】\033[0m')
                if response.lower() == 'y':
                    print('\033[94mStarting fresh.\033[0m')
                    load_path_old = load_path / f'{model.__class__.__name__}_checkpoint_old.pth'
                    load_path_full.rename(load_path_old)
                    return None
                else:
                    print('\033[93mAborting.\033[0m')
                    sys.exit()
            if use_saved_weights is not True:
                model.load_state_dict(checkpoint['model_state_dict'], strict=strict_load)
            elif last_epoch >= 3:
                print(
                    f'[\033[93mWarning\033[0m] Using custom weights instead of checkpoint weights may lead to instability in training. Use last saved best weights if still doing so.')
            if optimizer:
                if strict_load:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    self.optimizer_unmatch_load(optimizer, checkpoint['optimizer_state_dict'], verbose=True)

            return checkpoint

        print('\033[94mNo checkpoint found. Starting fresh.\033[0m') if verbose else None
        return None

    def end_save(self, model, optimizer, checkpoint):
        """
        Save the final model and optimizer state.
        """
        if checkpoint is not None:
            self.save_checkpoint(model, optimizer, self.config['train']['num_epochs'], checkpoint['min_loss'],
                                 checkpoint['max_accuracy'])

    def weights_unmatch_load(self, model, load_path):
        """
        Load weights into the model and handle mismatched keys.
        """
        old_state_dict = torch.load(load_path, weights_only=True, map_location=self.device)
        new_state_dict = model.state_dict()

        matched_weights = {
            k: v for k, v in old_state_dict.items()
            if k in new_state_dict and new_state_dict[k].shape == v.shape
        }
        new_state_dict.update(matched_weights)
        print(f"\n\033[94mLoading weights from {load_path}\033[0m")
        model.load_state_dict(new_state_dict, strict=False)

    def optimizer_unmatch_load(self, optimizer, old_state_dict, verbose=True):
        new_state_dict = optimizer.state_dict()

        old_state = old_state_dict.get('state', {})
        new_state = new_state_dict['state']
        matched_state = {k: v for k, v in old_state.items() if k in new_state}
        new_state.update(matched_state)

        if verbose:
            missing_state_keys = [k for k in new_state if k not in old_state]
            unexpected_state_keys = [k for k in old_state if k not in new_state]

            old_param_groups_cnt = len(old_state_dict.get('param_groups', []))
            new_param_groups_cnt = len(new_state_dict['param_groups'])
            param_groups_mismatch = old_param_groups_cnt != new_param_groups_cnt

            if missing_state_keys:
                print(f'[\033[93mWarning\033[0m] Missing optimizer state keys: {missing_state_keys}')
            if unexpected_state_keys:
                print(f'[\033[93mWarning\033[0m] Unexpected optimizer state keys: {unexpected_state_keys}')
            if param_groups_mismatch:
                print(
                    f'[\033[93mWarning\033[0m] Param groups count mismatch. Checkpoint has {old_param_groups_cnt}, current has {new_param_groups_cnt}. Using current structure.')

        optimizer.load_state_dict(new_state_dict)


class BuilderManager:
    def __init__(self):
        self.config_path = ModelSettings.config_path
        self.settings = ModelSettings(self.config_path)
        self.device = ModelSettings.device
        self.config = ModelSettings.load_config()
        self.object_string_parser = ModelSettings.object_string_parser
        self.flags = self.config['flags']
        self.data_dir_train = self.config['path']['data_dir_train']
        self.pos_dir_train = self.config['path']['pos_dir_train']
        self.neg_dir_train = self.config['path']['neg_dir_train']
        self.data_dir_test = self.config['path']['data_dir_test']
        self.pos_dir_test = self.config['path']['pos_dir_test']
        self.neg_dir_test = self.config['path']['neg_dir_test']
        self.train_csv = self.config['path']['train_csv']
        self.test_csv = self.config['path']['test_csv']
        self.data_dir_pred = self.config['path']['data_dir_pred']
        self.pred_csv = self.config['path']['pred_csv']
        self.test_output_dir = self.config['path']['test_output_dir']
        self.pred_output_dir = self.config['path']['pred_output_dir']
        self.image_size = self.config['data']['image_size']
        self.num_samples = self.config['data']['num_samples']
        self.num_val_samples = self.config['data']['num_val_samples']
        self.num_workers = self.config['data']['num_workers']
        self.augment_mode = self.config['data']['augment_mode']
        self.color_jitter = self.config['data']['color_jitter']
        self.add_noise = self.config['data']['add_noise']
        self.adaptation_mode = self.config['data']['adaptation_mode']
        self.channel_expansion_mode = self.config['data']['channel_expansion_mode']
        self.mix_channels = self.config['data']['mix_channels']
        self.csv_samples_catalog_reader = self.config['data']['csv_samples_catalog_reader']
        self.norm = self.config['data']['norm']
        self.update_mean_std = self.config['data']['update_mean_std']
        self.mean = self.config['data']['mean']
        self.std = self.config['data']['std']
        self.train_pos_label = self.config['data']['train_pos_label']
        self.model_config = self.config['model']
        self.batch_size = self.config['train']['batch_size']
        self.learning_rate = self.config['train']['learning_rate']
        self.num_epochs = self.config['train']['num_epochs']
        self.save_interval = self.config['train']['save_interval']

    def data_builder(self, test_mode=False, pred_mode=False, **kwargs):
        dataset, t_p_dataset = None, None
        hide_raw = kwargs.get('hide_process', False)
        seed = 42
        hide_main = test_mode or pred_mode
        stdout_controller = Interface.StdoutController()
        if test_mode:
            pos_dir = self.pos_dir_test
            neg_dir = self.neg_dir_test
            csv_file = self.test_csv
            data_dir = self.data_dir_test
        elif pred_mode:
            pos_dir = None
            neg_dir = None
            csv_file = self.pred_csv
            data_dir = self.data_dir_pred
        else:
            pos_dir = self.pos_dir_train
            neg_dir = self.neg_dir_train
            csv_file = self.train_csv
            data_dir = self.data_dir_train
        if hide_main:
            stdout_controller.stdout_block()
        dataset = ImageDataset(img_size=self.image_size, data_dir=data_dir, pos_dir=pos_dir, neg_dir=neg_dir,
                               csv_file=csv_file, num=self.num_samples, augment_mode=self.augment_mode,
                               color_jitter=self.color_jitter, add_noise=self.add_noise,
                               adaptation_mode=self.adaptation_mode, channel_expansion_mode=self.channel_expansion_mode,
                               mix_channels=self.mix_channels,
                               csv_samples_catalog_reader=self.csv_samples_catalog_reader, predicting_mode=pred_mode,
                               pos_label=self.train_pos_label, norm=self.norm, update_mean_std=self.update_mean_std,
                               mean=self.mean, std=self.std)
        if hide_main:
            stdout_controller.stdout_re()
        if test_mode:
            t_p_dataset = dataset.get_raw_dataset(hide_process=hide_raw, test=True)
        elif pred_mode:
            t_p_dataset = dataset.get_raw_dataset(hide_process=hide_raw, pred=True)
        if test_mode or pred_mode:
            return DataLoader(t_p_dataset, batch_size=self.batch_size, shuffle=False,
                              num_workers=self.num_workers, pin_memory=True)

        dataset_train, dataset_validation = self._fixed_split(dataset, self.num_val_samples, seed)[0], \
            self._fixed_split(dataset.get_raw_dataset(), self.num_val_samples, seed, True)[1]

        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers, pin_memory=True)

        dataloader_validation = DataLoader(dataset_validation, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers, pin_memory=True)

        return [dataloader_train, dataloader_validation]

    def model_builder(self, generate_summary=False):
        model_name = self.model_config['model_name']
        model_params = self.object_string_parser(self.model_config['model_params'], Mapping.STR_OBJECT_MAP)

        model = self._create_model(model_name, model_params, self.device)
        if generate_summary:
            summary(model, input_size=(
                self.config['train']['batch_size'], self.config['data']['input_channels'],
                self.config['data']['image_size'], self.config['data']['image_size']))
        return model

    def _get_model_class(self, model_name):
        if model_name not in Mapping.MODEL_MAPPING:
            raise ValueError(
                f"\033[93mWarning: Model {model_name} is not supported.\033[0m")

        return Mapping.MODEL_MAPPING[model_name]

    def _create_model(self, model_name, model_params, device):
        model_class = self._get_model_class(model_name)

        model = model_class(**model_params).to(device)
        return model

    @staticmethod
    def _fixed_split(dataset, val_size, seed=None, ignore_warning=False):

        dataset_size = len(dataset)

        if dataset_size < val_size:
            val_size = max(1, dataset_size // 5)
            print(
                f"[\033[93mWarning\033[0m] Validation size is too large. Adjusted to \033[93m{val_size}\033[0m (20% of dataset).") if not ignore_warning else None

        # Generate indices and optionally shuffle with a fixed seed
        indices = list(range(dataset_size))
        if seed is not None:
            np.random.seed(seed)
            np.random.shuffle(indices)

        train_size = dataset_size - val_size
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        dataset_train = Subset(dataset, train_indices)
        dataset_validation = Subset(dataset, val_indices)

        return dataset_train, dataset_validation

    @staticmethod
    def get_process_bar(config, stage, dataloader):
        """
        Configure the progress bar for a specific stage of training.

        :param config: Configuration dictionary loaded from the YAML file
        :param stage: Name of the stage of training (e.g. 'train', 'validation')
        :param dataloader: Dataloader for the specific stage of training
        :return: A progress bar object or None if not enabled in the configuration
        """
        process_bar_config = config['train']['process_bar']
        enable_self_defined_process_bar = process_bar_config.get('enabled', True)
        colour = process_bar_config[stage].get('colour')
        desc = process_bar_config[stage].get('desc', '')

        if enable_self_defined_process_bar:
            if desc:
                return tqdm(enumerate(dataloader), total=len(dataloader), desc=desc, colour=colour)
            else:
                return tqdm(enumerate(dataloader), total=len(dataloader), colour=colour)
        else:
            return None


class TrainingPipelineBuilder:
    def __init__(self):
        self.config = ModelSettings.load_config()
        self.config_stem = str(Path(ModelSettings.config_name).stem)
        self.log_dir = Path(self.config['path']['log_dir'])
        self.epoch_log_path = self.log_dir / f'epoch_logs/{self.config_stem}_log.csv'
        self.test_log_path = self.log_dir / f'test_logs/{self.config_stem}_log.txt'
        self.device = ModelSettings.device

        self.loss_map = Mapping.LOSS_MAP

        self.optimizer_map = Mapping.OPTIMIZER_MAP

        self.scheduler_map = Mapping.SCHEDULER_MAP

    def criterion_builder(self, verbose=True):
        loss_function_name = self.config['train']['loss_function']
        if loss_function_name in self.loss_map:
            print(f"Loss function: \033[94m{loss_function_name}\033[0m") if verbose else None
            return self.loss_map[loss_function_name]().to(self.device)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function_name}")

    def optimizer_builder(self, model, verbose=True):
        optimizer_name = self.config['train']['optimizer']
        learning_rate = self.config['train']['learning_rate']
        weight_decay = self.config['train']['weight_decay']

        if optimizer_name in self.optimizer_map:
            print(f"Optimizer: \033[94m{optimizer_name}\033[0m") if verbose else None
            return self.optimizer_map[optimizer_name](model.parameters(), lr=learning_rate,
                                                      weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def lr_scheduler_builder(self, optimizer, verbose=True):
        scheduler_config = self.config['train']['scheduler']
        scheduler_class_name = scheduler_config['class']
        scheduler_kwargs = None

        if scheduler_class_name in self.scheduler_map:
            print(f"Scheduler: \033[94m{scheduler_class_name}\033[0m") if verbose else None
            scheduler_class = self.scheduler_map[scheduler_class_name]
            if scheduler_class_name == 'CosineAnnealingLR':
                scheduler_kwargs = {
                    'T_max': scheduler_config['T_max'],
                    'eta_min': scheduler_config['eta_min']
                }
            return WarmupSchedulerWrapper(
                optimizer=optimizer,
                scheduler_class=scheduler_class,
                scheduler_kwargs=scheduler_kwargs,
                warmup_enabled=scheduler_config['warmup_enabled'],
                warmup_steps=scheduler_config['warmup_steps'],
                base_lr=self.config['train']['learning_rate']
            )
        else:
            raise ValueError(f"Unsupported scheduler class: {scheduler_class_name}")

    def setup_logging(self):
        """Create a CSV log file and write the header"""
        log_file_path = self.epoch_log_path
        log_dir = log_file_path.parent

        log_dir.mkdir(parents=True, exist_ok=True)

        if not log_file_path.exists():
            with open(log_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'val_loss', 'val_accuracy', 'train_loss', 'train_accuracy'])
                print(f"Log file (new created): \033[94m{log_file_path}\033[0m")
        else:
            print(f"Log file (using existed): \033[94m{log_file_path}\033[0m")

    def log_epoch(self, epoch, val_loss, val_accuracy, train_loss, train_accuracy):
        """Append or update epoch results to the CSV file"""
        log_file = self.epoch_log_path
        columns = ['epoch', 'val_loss', 'val_accuracy', 'train_loss', 'train_accuracy']

        new_row = {
            'epoch': epoch,
            'val_loss': f"{val_loss:.6f}" if val_loss is not None else "",
            'val_accuracy': f"{val_accuracy:.6f}" if val_accuracy is not None else "",
            'train_loss': f"{train_loss:.6f}" if train_loss is not None else "",
            'train_accuracy': f"{train_accuracy:.6f}" if train_accuracy is not None else ""
        }

        if log_file.exists():
            df = pd.read_csv(log_file)

            df = df[df['epoch'] != epoch]

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df = df.sort_values(by='epoch')
        else:
            df = pd.DataFrame([new_row], columns=columns)

        df.to_csv(log_file, index=False)

    def log_test_result(self, best_threshold, best_fpr, best_tpr, max_j, accuracy, loss, criterion_name, auc):
        """Write the test results to a log file (overwrite)"""
        log_file = self.test_log_path
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(
                f"Threshold(Youden): {best_threshold:.4f}, FPR: {best_fpr:.4f}, TPR: {best_tpr:.4f}, Max_J: {max_j:.4f}\n")
            f.write(f"Result: Accuracy: {accuracy:.4f}, Loss ({criterion_name}): {loss:.4f}, AUC: {auc:.4f}\n")

    def read_test_log(self, log_path=None):
        """Read the most recent test result from a log file (returns dict with None for missing values)"""
        log_path = self.test_log_path if log_path is None else Path(log_path)

        default_result = {
            'best_threshold': None,
            'best_fpr': None,
            'best_tpr': None,
            'max_j': None,
            'accuracy': None,
            'loss': None,
            'criterion': None,
            'auc': None
        }

        if not log_path.exists():
            print(f"[\033[93mWarning\033[0m] Log file not found: \033[93m{log_path}\033[0m")
            return default_result

        try:
            with open(log_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            if len(lines) < 2:
                print(f"[\033[93mWarning\033[0m] Incomplete log file: \033[93m{log_path}\033[0m")
                return default_result

            line1 = lines[-2]
            line2 = lines[-1]

            best_threshold = best_fpr = best_tpr = max_j = None
            if line1.startswith("Threshold") or line1.startswith("Threshold(Youden)"):
                parts1 = {kv.split(":")[0].strip(): kv.split(":")[1].strip()
                          for kv in line1.split(",") if ":" in kv}

                def try_float(key):
                    try:
                        return float(parts1.get(key, None))
                    except:
                        return None

                best_threshold = try_float("Threshold(Youden)")
                best_fpr = try_float("FPR")
                best_tpr = try_float("TPR")
                max_j = try_float("Max_J")

            accuracy = loss_value = auc = None
            criterion_name = None
            if line2.startswith("Result:"):
                values2 = line2.split(":", 1)[1]
                parts2 = [x.strip() for x in values2.split(",")]

                for part in parts2:
                    if part.startswith("Accuracy"):
                        try:
                            accuracy = float(part.split(":")[1].strip())
                        except:
                            accuracy = None
                    elif part.startswith("Loss"):
                        try:
                            loss_part = part.split(":", 1)
                            loss_value = float(loss_part[1].strip()) if len(loss_part) > 1 else None
                            criterion_name = part.split('(')[-1].split(')')[0].strip()
                        except:
                            loss_value = criterion_name = None
                    elif part.startswith("AUC"):
                        try:
                            auc = float(part.split(":")[1].strip())
                        except:
                            auc = None

            return {
                'best_threshold': best_threshold,
                'best_fpr': best_fpr,
                'best_tpr': best_tpr,
                'max_j': max_j,
                'accuracy': accuracy,
                'loss': loss_value,
                'criterion': criterion_name,
                'auc': auc
            }

        except Exception as e:
            print(f"[\033[91mError\033[0m] Failed to parse test log: {e}")
            return default_result
