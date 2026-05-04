import argparse
import builtins
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Callable

import torch
import torch.optim as optim
import torchvision
import yaml
from lion_pytorch import Lion
from tabulate import tabulate
from torch import nn
from torch.optim.lr_scheduler import *
from yaml.representer import SafeRepresenter

from model.CLFTNet import CLFTNet
from model.CLFTNet_casa import CLFTNetCaSa
from model.CLFTSwin import CLFTSwinTransformer
from model.DemiLensNet import DemiLensNet
from model.DemiLensNetAblation import DemiLensNetAblation
from model.DemiLensNet_dev import DemiLensNetDev
from model.DemiLensesSwin import DemiLensesSwin
from model.QuantumSwin import QuantumSwinTransformer
from model.ResNet import ResNet
from model.ViT import ViT
from model.cswin import CSWinTransformer
from model.swin_mlp import SwinMLP
from model.swin_transformer_moe import SwinTransformerMoE
from model.swin_transformer_v2 import SwinTransformerV2


@dataclass
class ModelSettings:
    config_name: str = 'default.yaml'
    config_dir: str = './config'
    config_path: str = str(Path(config_dir) / config_name)
    config: dict = field(init=False)
    _disk_config: dict = field(init=False)
    path_config: dict = field(init=False)
    data_config: dict = field(init=False)
    model_config: dict = field(init=False)
    train_config: dict = field(init=False)
    flags: dict = field(init=False)
    user_config_lib: list = None

    @classmethod
    def set_device(cls, device_index=0, verbose=False, info_verbose=False):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if isinstance(device_index, int):
                device_index = [device_index]
            device_index = [i % num_gpus for i in device_index]
            device = [torch.device(f"cuda:{i}") for i in device_index]
            device_name = [torch.cuda.get_device_name(i) for i in device_index]

            if verbose:
                print('\033[94mGPU detected!\033[0m')
                print('Num GPUs Available:', num_gpus)
                for i, name in zip(device_index, device_name):
                    print(f"Running on GPU: No.{i} \033[94m({name})\033[0m")

            if info_verbose:
                cls.print_gpu(num_gpus, device_index)

            if verbose or info_verbose:
                for i, name in zip(device_index, device_name):
                    cls.check_gpu_memory(i, name)

        else:
            device = [torch.device("cpu")]
            if verbose:
                print(f"Running on the CPU")
            if info_verbose:
                cls.print_cpu()

        cls.device = device[0] if len(device) == 1 else device

        return cls.device

    @classmethod
    def print_gpu(cls, num_gpu, used_gpu, show_all=False):
        headers = ["Index", "Name", "Status", "Compute Capability", "Total Memory (MB)", "Allocated Memory (MB)",
                   "Cached Memory (MB)", "Used Memory (MB)"] if show_all else [
            "Index", "Name", "Status", "Compute Capability", "Used Memory (MB)"]
        table_data = []

        for i in range(num_gpu):
            name = torch.cuda.get_device_name(i)
            capability = f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}"
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024 ** 2)
            cached_memory = torch.cuda.memory_reserved(i) / (1024 ** 2)

            used_memory, _ = cls._nvidia_smi_memory_reader(i)

            status = "[\033[92m√\033[0m] \033[94mIn Use\033[0m" if i in used_gpu else "[\033[91m×\033[0m] Idle"

            table_data.append(
                [i, name, status, capability, f"{total_memory:.2f}", f"{allocated_memory:.2f}", f"{cached_memory:.2f}",
                 f"{used_memory:.2f}"] if show_all else [i, name, status, capability, f"{used_memory:.2f}"])

        print("\033[94m-- GPU Info --\033[0m")
        print(tabulate(table_data, headers=headers, tablefmt="rounded_grid"))

    @staticmethod
    def print_cpu():
        cpu_count = os.cpu_count()
        cpu_info = platform.processor()
        cpu_architecture = platform.architecture()[0]

        headers = ["Processor", "Number of Cores", "Architecture"]
        table_data = [[cpu_info, cpu_count, cpu_architecture]]

        print("\033[94m-- CPU Info --\033[0m")
        print(tabulate(table_data, headers=headers, tablefmt="rounded_grid"))

    @classmethod
    def check_gpu_memory(cls, device_index, device_name, threshold=0.30):
        used_memory, total_memory = cls._nvidia_smi_memory_reader(device_index)

        memory_usage_percentage = used_memory / total_memory

        if memory_usage_percentage > threshold:
            print(
                f"\033[93mWarning: {device_name} (GPU {device_index}) has high memory usage ({memory_usage_percentage * 100:.2f}%). Make sure you are using the right device.\033[0m")
        else:
            print(
                f"{device_name} (GPU {device_index}) memory usage is normal \033[94m({memory_usage_percentage * 100:.2f}%)\033[0m.")

    @staticmethod
    def _nvidia_smi_memory_reader(gpu_index):
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        memory_info = result.stdout.splitlines()[gpu_index]
        used_memory, total_memory = map(int, memory_info.split(','))
        return used_memory, total_memory

    @classmethod
    def init_config(cls, config_name=config_name, use_default=False, config_dir='default', full_path=False):
        """
        initialize config
        """
        if full_path is not True:
            config_dir = cls.config_dir if config_dir == 'default' else config_dir
            cls.config_path = str(Path(config_dir) / config_name)
        else:
            cls.config_path = config_name
            config_name = Path(config_name).name

        load_path = cls.config_path

        print(f"Loading config from {load_path}...")
        if not os.path.exists(load_path):
            if use_default:
                print(f"\033[93mWarning: Config file {load_path} does not exist, using default config.\033[0m")
                cls.config_name = 'default.yaml'
            else:
                raise FileNotFoundError(f"Config file {load_path} does not exist.")

        cls.config = cls._safe_init()
        cls.config_name = config_name
        cls.last_saved_config = cls.load_config()

    @classmethod
    def _safe_init(cls):
        with open(cls.config_path) as f:
            return yaml.safe_load(f)

    @classmethod
    def load_config(cls):
        """
        Load config, update the sub-parameter tree (used without assignment) or for retrieving the current configuration.
        """

        cls.path_config = cls.config['path']
        cls.data_config = cls.config['data']
        cls.model_config = cls.config['model']
        cls.train_config = cls.config['train']
        cls.flags = cls.config['flags']

        return cls.config

    @classmethod
    def save_config(cls, external_save=False, **kwargs):
        """
        Safely save the current config to file while preserving the original file structure and comments.
        """
        save_path = None

        if external_save:
            if 'config' in kwargs and 'path' in kwargs:
                full_config, save_path = kwargs['config'], kwargs['path']
            else:
                raise ValueError(
                    "Invalid arguments for external config save. Make sure you 'config' and 'path' are passed, or set 'external_save' to False.")
        else:
            model_config_str = cls._object_serializer(cls.model_config, Mapping.STR_OBJECT_MA_REVERSE)
            cls.last_saved_config = cls.load_config()

            full_config = {
                "flags": cls.flags,
                "path": cls.path_config,
                "data": cls.data_config,
                "train": cls.train_config,
                "model": model_config_str
            }
            save_path = cls.config_path

        temp_path = save_path + ".tmp"
        with open(temp_path, 'w') as f:
            yaml.dump(
                full_config, f,
                Dumper=cls.CustomDumper,
                indent=2,
                allow_unicode=True,
                sort_keys=False,
                explicit_start=True,
                default_style=None,
                width=1000
            )

        if os.path.exists(save_path):
            os.replace(temp_path, save_path)
        else:
            os.rename(temp_path, save_path)

    @classmethod
    def get_config_paths(cls, stored=False):
        import os
        import re
        if stored:
            if cls.user_config_lib is not None:
                return cls.user_config_lib

        paths = []

        print(
            "\033[94mPlease input paths of the config file, each config refers to a trained model.\033[0m")
        while True:
            path = input(
                "Path of the config file (press \033[96mEnter\033[0m to finish): ")
            if path == '':
                if not paths:
                    print("[\033[91mError\033[0m] No config file path is provided.")
                    continue
                break
            if re.search(r'[<>:"|?*\x00-\x1F]', path):
                print("[\033[91mError\033[0m] Invalid filename.")
                continue
            if not os.path.exists(path):
                print(
                    f"[\033[91mError\033[0m] Config file \033[91m{path}\033[0m does not exist. (Is the content included?)")
                continue
            paths.append(path)

        print("The following config files will be used:")
        for path in paths:
            print(f"  - \033[96m{path}\033[0m")

        confirmation = input("Is the config info correct? (y/n)：")

        if confirmation.lower() in ['y', 'yes', '']:
            cls.user_config_lib = paths if stored else None
            return paths
        else:
            time.sleep(1)
            print()
            return cls.get_config_paths()

    @classmethod
    def setup_training(cls, auto_apply=True):
        """
        Dynamic configuration update executor, handling updates of different levels based on priority.
        """
        diffs = cls._config_update_checker()

        if diffs:
            print("\n\033[93mTrying to apply config updates...\033[0m")
            for path, change_type, old_val, new_val in diffs:
                print(f"- {path}: {change_type}, {old_val} ==> {new_val}")

            if auto_apply or input("\nApply changes? (y/n): ").lower() == 'y':
                cls.config = cls._disk_config
                cls.load_config()

                need_model_rebuild = any(
                    Mapping.PARAM_MAP.get(path.split('.', 1)[-1], 'unknown') == 'model_rebuild'
                    for path, _, _, _ in diffs
                )
                need_data_rebuild = any(
                    Mapping.PARAM_MAP.get(path.split('.', 1)[-1], 'unknown') == 'dataset_rebuild'
                    for path, _, _, _ in diffs
                )

                if need_model_rebuild:
                    cls.flags['model_rebuild'] = True
                if need_data_rebuild:
                    cls.flags['dataset_rebuild'] = True
        update_handlers = [
            ('model_rebuild', cls._model_rebuild_update),
            ('dataset_rebuild', cls._dataset_rebuild_update),
            ('params_update', cls._normal_update)
        ]

        for flag_name, handler in update_handlers:
            if cls.flags.get(flag_name, False):
                handler()
                cls.flags[flag_name] = False

        if cls.last_saved_config != cls.config:
            cls.save_config()

    @classmethod
    def _config_update_checker(cls):
        last_config = cls.load_config()
        new_config = cls._safe_init()

        diffs = cls._config_diff_finder(new_config, last_config)

        cls._disk_config = new_config

        return diffs

    @classmethod
    def _config_diff_finder(cls, old, new, path=""):
        diffs = []
        for key in new:
            new_path = f"{path}.{key}" if path else key
            if key not in old:
                diffs.append((new_path, "ADDED", None, new[key]))
                continue

            old_val = old[key]
            new_val = new[key]

            if isinstance(old_val, dict) and isinstance(new_val, dict):
                diffs.extend(cls._config_diff_finder(old_val, new_val, new_path))
            elif old_val != new_val:
                diffs.append((new_path, "MODIFIED", old_val, new_val))

        for key in old:
            if key not in new:
                new_path = f"{path}.{key}" if path else key
                diffs.append((new_path, "REMOVED", old[key], None))

        return diffs

    @classmethod
    def _normal_update(cls):
        normal_params = [
            param for param, param_type in Mapping.PARAM_MAP.items()
            if param_type == 'normal' and not param.startswith('flags.')
        ]
        updates = {
            param: cls._get_nested_config(param)
            for param in normal_params
        }
        cls._config_update_dealer(updates, 'params_update')

    @classmethod
    def _model_rebuild_update(cls):
        model_params = [
            param for param, param_type in Mapping.PARAM_MAP.items()
            if param_type in ('model_rebuild', 'normal') and not param.startswith('flags.')
        ]
        updates = {
            param: cls._get_nested_config(param)
            for param in model_params
        }
        cls._config_update_dealer(updates, 'model_rebuild')

    @classmethod
    def _dataset_rebuild_update(cls):
        data_params = [
            param for param, param_type in Mapping.PARAM_MAP.items()
            if param_type in ('dataset_rebuild', 'normal') and not param.startswith('flags.')
        ]
        updates = {
            param: cls._get_nested_config(param)
            for param in data_params
        }
        cls._config_update_dealer(updates, 'dataset_rebuild')

    @classmethod
    def _get_nested_config(cls, param_path):
        """
        Safely retrieve nested configuration values.
        """
        keys = param_path.split('.')
        current = cls.config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                raise ValueError(f"Invalid config path: {param_path}")
        return current

    @classmethod
    def _config_update_dealer(cls, config_updates, flag):
        """
        update config with new values
        :param config_updates: a dictionary of config updates
        :param flag: the flag that enables the update
        :return: the updated config
        """

        def _get_required_flag(path):
            return {
                'normal': 'params_update',
                'dataset_rebuild': 'dataset_rebuild',
                'model_rebuild': 'model_rebuild'
            }.get(path, 'unknown')

        def _update_nested_param(config, path, value):
            keys = path.split('.')
            current = config
            try:
                for key in keys[:-1]:
                    current = current[key]
                current_key = keys[-1]
                if current_key not in current:
                    return False
                if current[current_key] == value:
                    return False
                current[current_key] = value
                return True
            except (KeyError, TypeError):
                return False

        update_rules = {
            'params_update': ['normal'],
            'dataset_rebuild': ['dataset_rebuild', 'normal'],
            'model_rebuild': ['model_rebuild', 'normal']
        }

        if not isinstance(config_updates, dict):
            raise TypeError("config_updates must be a dictionary")

        if flag not in update_rules or not cls.flags.get(flag, False):
            print(f"\033[93mWarning: Flag {flag} is not enabled or invalid.\033[0m")
            return cls.config

        allowed_types = update_rules[flag]
        updated_params = []

        seen_params = set()
        for param_path, new_value in config_updates.items():
            if param_path in seen_params:
                continue
            seen_params.add(param_path)
            param_type = Mapping.PARAM_MAP.get(param_path, 'unknown')

            if param_type == 'static':
                print(f"\033[93mWarning: Static param {param_path} cannot be modified during training.\033[0m")
                continue

            if param_type not in allowed_types:
                required_flag = _get_required_flag(param_type)
                print(
                    f"\033[93mWarning: {param_path} requires {param_type} update. Need to enable '{required_flag}' flag.\033[0m")
                continue

            if _update_nested_param(cls.config, param_path, new_value):
                updated_params.append(param_path)
            else:
                print(f"\033[93mWarning: Invalid parameter path {param_path}.\033[0m")

        if updated_params:
            print("\n\033[92mSuccessfully updated parameters:\033[0m")
            for p in updated_params:
                print(f"- {p}")

        return cls.config

    @staticmethod
    def object_string_parser(config, matching_map):
        """
        Parse object strings in the configuration and convert them to actual Python objects.

        :param config: Configuration dictionary loaded from the YAML file
        :param matching_map: A mapping dictionary from strings to actual Python classes
        :return: Updated configuration dictionary
        """
        for param, param_value in config.items():
            if isinstance(param_value, str):
                if '.' in param_value:
                    module_path, _, class_name = param_value.rpartition('.')
                    try:
                        module = __import__(module_path, fromlist=[class_name])
                        config[param] = getattr(module, class_name)
                        continue
                    except (ImportError, AttributeError):
                        pass

                class_obj = matching_map.get(param_value)
                if class_obj is not None:
                    config[param] = class_obj

            elif isinstance(param_value, dict):
                ModelSettings.object_string_parser(param_value, matching_map)

        return config

    @staticmethod
    def _object_serializer(config, matching_map):
        """
        Serialize Python objects in the configuration to string representations.

        :param config: Configuration dictionary loaded from the YAML file
        :param matching_map: A mapping dictionary from Python classes to string representations
        :return: Updated configuration dictionary with string representations of Python objects
        """

        def convert_value(value):
            if isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(v) for v in value]
            elif isinstance(value, type) and value in matching_map:
                return matching_map[value]
            elif callable(value):
                return value.__name__ if hasattr(value, '__name__') else str(value)
            return value

        return convert_value(config)

    class CustomDumper(yaml.SafeDumper):
        """
        Custom YAML dumper that preserves the original file structure and comments.
        """

        def represent_sequence(self, tag, sequence, flow_style=None):
            return SafeRepresenter.represent_sequence(
                self, tag, sequence, flow_style=True
            )

        def represent_mapping(self, tag, mapping, flow_style=None):
            return SafeRepresenter.represent_mapping(
                self, tag, mapping, flow_style=False
            )


@dataclass
class Interface:
    executors: List["Interface.WorkflowExecutor"] = field(default_factory=list)

    @classmethod
    def _run_releaser(cls, name: str, auto_go: bool, time_wait: int = 15, pre_instr: any = None):
        if auto_go != -1:
            cls.divider_printer()
        if auto_go is False:
            input(f"Breakpoint for \033[94m<{name}>\033[0m, press \033[94mEnter\033[0m to execute...")
        elif auto_go is True:
            cls.countdown_pass(time_wait, name, 'workflow')
        if pre_instr is not None:
            ModelSettings.user_config_lib = ModelSettings.get_config_paths(
                stored=True) if pre_instr == 'get_usr_conf' else None
            print(Mapping.STAGE_DESCRIPTION_MAP[name])
        if auto_go != -1:
            time.sleep(0.5)

    @classmethod
    def _restart_dealer(cls, name: str, auto_restart: bool, e: Exception):
        print(f"\033[91mError: Failed to execute <{name}>: {e}\033[0m")
        if auto_restart:
            print(f"\033[93mAuto restarting <{name}>.\033[0m")
        else:
            input("\033[91mPress Enter to attempt restart...\033[0m")
            print(f"\033[93mRestarting: <{name}>.\033[0m")

    @staticmethod
    def divider_printer(width=100):
        try:
            width = os.get_terminal_size().columns
        except OSError:
            pass
        print("-" * width)

    @classmethod
    def countdown_pass(cls, time_wait, name=None, mode='workflow'):
        if sys.platform.startswith('win'):
            cls._countdown_pass_win(time_wait, name, mode)
        else:
            cls._countdown_pass_linux_else(time_wait, name, mode)
        print()

    @classmethod
    def _countdown_pass_win(cls, time_wait, name=None, mode='workflow'):
        import msvcrt
        for i in range(time_wait, 0, -1):
            if msvcrt.kbhit():
                msvcrt.getch()
                return
            cls._countdown_printer(i, name, mode)

    @classmethod
    def _countdown_pass_linux_else(cls, time_wait, name=None, mode='workflow'):
        import select
        import termios
        import tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            for i in range(time_wait, 0, -1):
                rlist, _, _ = select.select([sys.stdin], [], [], 0)
                if rlist:
                    sys.stdin.read(1)
                    return
                cls._countdown_printer(i, name, mode)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    @classmethod
    def _countdown_printer(cls, i, name=None, mode='workflow'):
        if mode == 'workflow' and name:
            print(f"\rBreakpoint for \033[94m<{name}>\033[0m, executing in \033[94m{i}\033[0m seconds...", end="",
                  flush=True)
        else:
            print(f"\r\033[94m{i}\033[0m seconds remaining... Press any key to skip...", end="", flush=True)
        time.sleep(1)

    @dataclass
    class WorkflowExecutor:
        name: str
        function: Callable
        args: tuple = field(default_factory=tuple)
        kwargs: dict = field(default_factory=dict)
        success: bool = False
        restart_count: int = 0

        def run(self, auto_go: bool, auto_restart: bool, *args, **kwargs):
            start_time_wait, max_restart_count = args if args else (15, 10)
            Interface._run_releaser(self.name, auto_go, time_wait=start_time_wait,
                                    **dict((key, kwargs[key]) for key in ['pre_instr'] if key in kwargs))
            try:
                self.function(*self.args, **self.kwargs)
                self.success = True
                print('\033[92mDone.\033[0m') if kwargs.get('show_end', True) else None
            except Exception as e:
                self.success = False
                self.restart_count += 1
                if self.restart_count >= max_restart_count and auto_restart:
                    raise RuntimeError(
                        f'\033[91mError: Failed to execute \033[94m<{self.name}>\033[91m, aborting.\033[0m')
                Interface._restart_dealer(self.name, auto_restart, e)
                self._restart(-1, auto_restart)

        def _restart(self, auto_go, auto_restart):
            if auto_restart is True:
                time.sleep(2)
            self.run(auto_go, auto_restart)

    class StdoutController:
        def __init__(self):
            self.standard_stdout = sys.stdout
            self.standard_input = builtins.input

        def stdout_block(self):
            builtins.input = self._fake_input
            sys.stdout = self.NullWriter()

        def stdout_re(self):
            sys.stdout = self.standard_stdout
            builtins.input = self.standard_input

        def _fake_input(self, prompt=None):
            return ''

        class NullWriter:
            def write(self, _):
                pass

            def flush(self):
                pass

            def writelines(self, lines):
                pass

            def isatty(self):
                return False

            def fileno(self):
                return 1

            def readable(self):
                return False

            def writable(self):
                return True

            def seekable(self):
                return False

            def close(self):
                pass

            def closed(self):
                return False


@dataclass(frozen=True)
class Mapping:
    STAGE_DESCRIPTION_MAP = {
        'Init Config': '\033[38;5;22m\033[47mInitializing config...\033[0m',
        'Set Device': '\033[38;5;22m\033[47mSetting up device...\033[0m',
        'Data Preparation': '\033[38;5;22m\033[47mPreparing data...\033[0m',
        'Build Model': '\033[38;5;22m\033[47mBuilding model...\033[0m',
        'Build Train Pipeline': '\033[38;5;22m\033[47mBuilding train pipeline...\033[0m',
        'Train Model': '\033[38;5;22m\033[47mStarting training...\033[0m',
        'Test Model': '\033[38;5;22m\033[47mTesting model...\033[0m',
        'Analyze Results': '\033[38;5;22m\033[47mAnalyzing results...\033[0m',
        'Model Prediction': '\033[38;5;22m\033[47mStarting prediction...\033[0m'
    }

    PARAM_MAP = {
        # flags
        'flags.model_rebuild': 'model_rebuild',
        'flags.dataset_rebuild': 'dataset_rebuild',
        'flags.params_update': 'normal',

        # paths
        'path.data_dir_train': 'static',
        'path.pos_dir_train': 'static',
        'path.neg_dir_train': 'static',
        'path.train_csv': 'static',
        'path.data_dir_test': 'static',
        'path.pos_dir_test': 'static',
        'path.neg_dir_test': 'static',
        'path.test_csv': 'static',
        'path.config_path': 'static',
        'path.weights_path': 'static',
        'path.checkpoint_dir': 'static',

        # data
        'data.input_channels': 'dataset_rebuild',
        'data.image_size': 'dataset_rebuild',
        'data.num_workers': 'normal',
        'data.num_samples': 'normal',
        'data.num_val_samples': 'normal',
        'data.train_pos_label': 'normal',
        'data.augment': 'normal',
        'data.color_jitter': 'normal',
        'data.add_noise': 'normal',
        'data.adaptation_mode': 'normal',
        'data.channel_expansion_mode': 'dataset_rebuild',
        'data.mix_channels': 'dataset_rebuild',
        'data.csv_samples_catalog_reader': 'dataset_rebuild',
        'data.update_mean_std': 'dataset_rebuild',
        'data.norm': 'normal',
        'data.std': 'normal',
        'data.mean': 'normal',

        # train
        'train.optimizer': 'static',
        'train.loss_function': 'static',
        'train.scheduler.class': 'model_rebuild',
        'train.scheduler.T_max': 'normal',
        'train.scheduler.eta_min': 'normal',
        'train.scheduler.warmup_enabled': 'normal',
        'train.scheduler.warmup_steps': 'normal',
        'train.learning_rate': 'normal',
        'train.batch_size': 'normal',
        'train.num_epochs': 'normal',
        'train.weight_decay': 'normal',
        'train.save_interval': 'normal',
        'train.self_evaluate_interval': 'normal',
        'train.process_bar.enabled': 'static',
        'train.process_bar.train.colour': 'static',
        'train.process_bar.train.desc': 'static',
        'train.process_bar.evaluate.colour': 'static',
        'train.process_bar.evaluate.desc': 'static',
        'train.process_bar.self_evaluate.colour': 'static',
        'train.process_bar.self_evaluate.desc': 'static',

        # model
        'model.model_name': 'static',
        'model.model_params.img_size': 'model_rebuild',
        'model.model_params.patch_size': 'model_rebuild',
        'model.model_params.in_chans': 'model_rebuild',
        'model.model_params.num_classes': 'model_rebuild',
        'model.model_params.embed_dim': 'model_rebuild',
        'model.model_params.depths': 'model_rebuild',
        'model.model_params.num_heads': 'model_rebuild',
        'model.model_params.window_size': 'model_rebuild',
        'model.model_params.mlp_ratio': 'normal',
        'model.model_params.qkv_bias': 'normal',
        'model.model_params.drop_rate': 'normal',
        'model.model_params.attn_drop_rate': 'normal',
        'model.model_params.drop_path_rate': 'normal',
        'model.model_params.norm_layer': 'model_rebuild',
        'model.model_params.ape': 'model_rebuild',
        'model.model_params.patch_norm': 'model_rebuild',
        'model.model_params.use_checkpoint': 'static',
    }

    STR_OBJECT_MAP = {
        "Linear": nn.Linear,
        "Conv2d": nn.Conv2d,
        "BatchNorm2d": nn.BatchNorm2d,
        "ReLU": nn.ReLU,
        "MaxPool2d": nn.MaxPool2d,
        "AvgPool2d": nn.AvgPool2d,
        "Dropout": nn.Dropout,
        "Sequential": nn.Sequential,
        "ModuleList": nn.ModuleList,
        "ModuleDict": nn.ModuleDict,
        "Identity": nn.Identity,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
        "Softmax": nn.Softmax,
        "LogSoftmax": nn.LogSoftmax,
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
        "BCELoss": nn.BCELoss,
        "MSELoss": nn.MSELoss,
        "CrossEntropyLoss": nn.CrossEntropyLoss,
        "NLLLoss": nn.NLLLoss,
        "L1Loss": nn.L1Loss,
        "HuberLoss": nn.SmoothL1Loss,
        "PoissonNLLLoss": nn.PoissonNLLLoss,
        "KLLoss": nn.KLDivLoss,
        "MarginRankingLoss": nn.MarginRankingLoss,
        "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
        "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
        "TripletMarginLoss": nn.TripletMarginLoss,
        "CTCLoss": nn.CTCLoss,
        "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
        "AdaptiveMaxPool2d": nn.AdaptiveMaxPool2d,
        "Upsample": nn.Upsample,
        "UpsamplingBilinear2d": nn.UpsamplingBilinear2d,
        "UpsamplingNearest2d": nn.UpsamplingNearest2d,
        "ZeroPad2d": nn.ZeroPad2d,
        "ConstantPad1d": nn.ConstantPad1d,
        "ConstantPad2d": nn.ConstantPad2d,
        "ConstantPad3d": nn.ConstantPad3d,
        "ReplicationPad1d": nn.ReplicationPad1d,
        "ReplicationPad2d": nn.ReplicationPad2d,
        "ReplicationPad3d": nn.ReplicationPad3d,
        "GroupNorm": nn.GroupNorm,
        "InstanceNorm1d": nn.InstanceNorm1d,
        "InstanceNorm2d": nn.InstanceNorm2d,
        "InstanceNorm3d": nn.InstanceNorm3d,
        "LayerNorm": nn.LayerNorm,
        "LocalResponseNorm": nn.LocalResponseNorm,
        "CrossMapLRN2d": nn.CrossMapLRN2d,
        "Embedding": nn.Embedding,
        "EmbeddingBag": nn.EmbeddingBag,
        "RNN": nn.RNN,
        "LSTM": nn.LSTM,
        "GRU": nn.GRU,
        "RNNCell": nn.RNNCell,
        "LSTMCell": nn.LSTMCell,
        "GRUCell": nn.GRUCell,
        "PixelShuffle": nn.PixelShuffle,
    }

    STR_OBJECT_MA_REVERSE = {v: k for k, v in STR_OBJECT_MAP.items()}

    MODEL_MAPPING = {
        "vit": ViT,
        "swintransformerv2": SwinTransformerV2,
        "swin_mlp": SwinMLP,
        "swin_transformer_moe": SwinTransformerMoE,
        "resnet_specified": ResNet,
        "demilensesswin": DemiLensesSwin,
        "quantumswin": QuantumSwinTransformer,
        "clftswintransformer": CLFTSwinTransformer,
        "cswin": CSWinTransformer,
        "clftnet": CLFTNet,
        "clftfdia": DemiLensNetDev,
        "clftnetcasa": CLFTNetCaSa,
        "demilensnetdev": DemiLensNetDev,
        "demilensnet": DemiLensNet,
        "demilensnet_ablated": DemiLensNetAblation,
        "resnet101": torchvision.models.resnet101,
        "resnet152": torchvision.models.resnet152,
        "resnet34": torchvision.models.resnet34,
        "resnext50_32x4d": torchvision.models.resnext50_32x4d,
        "resnext101_32x8d": torchvision.models.resnext101_32x8d,
        "wide_resnet50_2": torchvision.models.wide_resnet50_2,
        "wide_resnet101_2": torchvision.models.wide_resnet101_2,
        "vgg16": torchvision.models.vgg16,
        "vgg19": torchvision.models.vgg19,
        "densenet121": torchvision.models.densenet121,
        "densenet161": torchvision.models.densenet161,
        "densenet169": torchvision.models.densenet169,
        "densenet201": torchvision.models.densenet201,
        "mobilenet_v2": torchvision.models.mobilenet_v2,
        "mobilenetv2": torchvision.models.mobilenet_v2,
        "mnasnet0_5": torchvision.models.mnasnet0_5,
        "mnasnet0_75": torchvision.models.mnasnet0_75,
        "mnasnet1_0": torchvision.models.mnasnet1_0,
        "mnasnet1_3": torchvision.models.mnasnet1_3,
        "shufflenet_v2_x0_5": torchvision.models.shufflenet_v2_x0_5,
        "shufflenet_v2_x1_0": torchvision.models.shufflenet_v2_x1_0,
        "shufflenet_v2_x1_5": torchvision.models.shufflenet_v2_x1_5,
        "shufflenet_v2_x2_0": torchvision.models.shufflenet_v2_x2_0,
        "squeezenet1_0": torchvision.models.squeezenet1_0,
        "squeezenet1_1": torchvision.models.squeezenet1_1,
        "alexnet": torchvision.models.alexnet,
        "googlenet": torchvision.models.googlenet,
        "inception_v3": torchvision.models.inception_v3,
        "inceptionv3": torchvision.models.inception_v3,
        "resnet50": torchvision.models.resnet50,
        "resnet18": torchvision.models.resnet18,
    }

    STR_OBJECT_MAP = {
        "Linear": nn.Linear,
        "Conv2d": nn.Conv2d,
        "BatchNorm2d": nn.BatchNorm2d,
        "ReLU": nn.ReLU,
        "MaxPool2d": nn.MaxPool2d,
        "AvgPool2d": nn.AvgPool2d,
        "Dropout": nn.Dropout,
        "Sequential": nn.Sequential,
        "ModuleList": nn.ModuleList,
        "ModuleDict": nn.ModuleDict,
        "Identity": nn.Identity,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
        "Softmax": nn.Softmax,
        "LogSoftmax": nn.LogSoftmax,
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
        "BCELoss": nn.BCELoss,
        "MSELoss": nn.MSELoss,
        "CrossEntropyLoss": nn.CrossEntropyLoss,
        "NLLLoss": nn.NLLLoss,
        "L1Loss": nn.L1Loss,
        "HuberLoss": nn.SmoothL1Loss,
        "PoissonNLLLoss": nn.PoissonNLLLoss,
        "KLLoss": nn.KLDivLoss,
        "MarginRankingLoss": nn.MarginRankingLoss,
        "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
        "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
        "TripletMarginLoss": nn.TripletMarginLoss,
        "CTCLoss": nn.CTCLoss,
        "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
        "AdaptiveMaxPool2d": nn.AdaptiveMaxPool2d,
        "Upsample": nn.Upsample,
        "UpsamplingBilinear2d": nn.UpsamplingBilinear2d,
        "UpsamplingNearest2d": nn.UpsamplingNearest2d,
        "ZeroPad2d": nn.ZeroPad2d,
        "ConstantPad1d": nn.ConstantPad1d,
        "ConstantPad2d": nn.ConstantPad2d,
        "ConstantPad3d": nn.ConstantPad3d,
        "ReplicationPad1d": nn.ReplicationPad1d,
        "ReplicationPad2d": nn.ReplicationPad2d,
        "ReplicationPad3d": nn.ReplicationPad3d,
        "GroupNorm": nn.GroupNorm,
        "InstanceNorm1d": nn.InstanceNorm1d,
        "InstanceNorm2d": nn.InstanceNorm2d,
        "InstanceNorm3d": nn.InstanceNorm3d,
        "LayerNorm": nn.LayerNorm,
        "LocalResponseNorm": nn.LocalResponseNorm,
        "CrossMapLRN2d": nn.CrossMapLRN2d,
        "Embedding": nn.Embedding,
        "EmbeddingBag": nn.EmbeddingBag,
        "RNN": nn.RNN,
        "LSTM": nn.LSTM,
        "GRU": nn.GRU,
        "RNNCell": nn.RNNCell,
        "LSTMCell": nn.LSTMCell,
        "GRUCell": nn.GRUCell,
        "PixelShuffle": nn.PixelShuffle,
    }

    LOSS_MAP = {
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,  # 二分类，Logits输出
        'BCELoss': nn.BCELoss,  # 二分类，Sigmoid输出
        'MSELoss': nn.MSELoss,  # 回归
        'CrossEntropyLoss': nn.CrossEntropyLoss,  # 多分类，Logits输出
        'NLLLoss': nn.NLLLoss,  # 多分类，Logits输出
        'L1Loss': nn.L1Loss,  # 回归
        'HuberLoss': nn.SmoothL1Loss,  # 回归
        'PoissonNLLLoss': nn.PoissonNLLLoss,  # 泊松回归
        'KLLoss': nn.KLDivLoss,  # KL散度
        'MarginRankingLoss': nn.MarginRankingLoss,  # 排名任务
        'MultiLabelSoftMarginLoss': nn.MultiLabelSoftMarginLoss,  # 多标签分类
        'CosineEmbeddingLoss': nn.CosineEmbeddingLoss,  # 余弦相似度
        'TripletMarginLoss': nn.TripletMarginLoss,  # 三元组损失
        'CTCLoss': nn.CTCLoss  # 用于序列到序列的任务，通常用于语音识别
    }

    OPTIMIZER_MAP = {
        'Adam': optim.Adam,
        'SGD': optim.SGD,
        'AdamW': optim.AdamW,
        'RMSprop': optim.RMSprop,
        'Adagrad': optim.Adagrad,
        'Adadelta': optim.Adadelta,
        'Lion': Lion
    }

    SCHEDULER_MAP = {
        'CosineAnnealingLR': CosineAnnealingLR,
        'StepLR': StepLR,
        'ExponentialLR': ExponentialLR,
        'LambdaLR': LambdaLR,
        'ReduceLROnPlateau': ReduceLROnPlateau,
        'CyclicLR': CyclicLR,
        'OneCycleLR': OneCycleLR
    }

    class ConfigSetup:
        """A class to manage the config file."""
        CONFIG_DIR = ModelSettings.config_dir
        DEFAULT_CONFIG = str(Path(CONFIG_DIR) / 'default.yaml')
        DEFAULT_CONFIG_BACKUP = BACKUP_CONFIG = {
            'flags': {'model_rebuild': False, 'dataset_rebuild': False, 'params_update': False},
            'path': {
                'config_dir': './config', 'data_dir_train': None, 'data_dir_test': None,
                'pos_dir_train': ['./data/train/positives/3band'],
                'neg_dir_train': ['./data/train/negatives/neg/3band', './data/train/negatives/LRGs/3band'],
                'train_csv': None, 'pos_dir_test': ['./data/test/positives/3band'],
                'neg_dir_test': ['./data/test/negatives/3band'], 'test_csv': None,
                'data_dir_pred': ['./data/test/test_real'], 'pred_csv': None, 'pred_output_dir': './result/pred',
                'pred_output_type': 'file', 'test_output_dir': './result/csv', 'plot_output_dir': './result/fig',
                'weights_dir': './weights', 'checkpoint_dir': './checkpoints', 'log_dir': './logs'
            },
            'data': {
                'input_channels': 3, 'image_size': 96, 'num_workers': 10, 'num_samples': None, 'num_val_samples': 8000,
                'train_pos_label': 1.0, 'augment_mode': 'full', 'color_jitter': False, 'add_noise': False,
                'adaptation_mode': 'padding', 'channel_expansion_mode': None, 'mix_channels': False,
                'csv_samples_catalog_reader': 'sample', 'norm': False, 'update_mean_std': False, 'mean': None,
                'std': None
            },
            'train': {
                'optimizer': 'AdamW', 'loss_function': 'BCEWithLogitsLoss',
                'scheduler': {'class': 'CosineAnnealingLR', 'T_max': 30, 'eta_min': 1.0e-11, 'warmup_enabled': True,
                              'warmup_steps': 10},
                'learning_rate': 0.0005, 'batch_size': 128, 'num_epochs': 1000, 'weight_decay': 1e-5,
                'save_interval': 1, 'self_evaluate_interval': 5,
                'process_bar': {
                    'enabled': True,
                    'train': {'colour': 'red', 'desc': 'Training'},
                    'evaluate': {'colour': 'blue', 'desc': 'Evaluating'},
                    'self-evaluate': {'colour': 'yellow', 'desc': 'Self-evaluating'}
                }
            },
            'model': {
                'model_name': '# add model here', 'model_params': '# add model parameters here'
            }
        }
        MODEL_PARAMS_MAP = {
            "vit": {
                'img_size': 96, 'patch_size': 4, 'in_channels': 3, 'embed_dim': 768, 'depth': 12, 'num_heads': 12,
                'mlp_ratio': 4.0, 'dropout': 0.0, 'attn_dropout': 0.0,
            },
            "swintransformerv2": {
                'img_size': 96, 'patch_size': 4, 'in_chans': 3, 'num_classes': 1, 'embed_dim': 96,
                'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24], 'window_size': 6, 'mlp_ratio': 4,
                'qkv_bias': True, 'drop_rate': 0., 'attn_drop_rate': 0., 'drop_path_rate': 0.,
                'norm_layer': "torch.nn.LayerNorm", 'ape': False, 'patch_norm': True, 'use_checkpoint': False
            },
            "resnet_specified": {
                'in_chans': 3, 'use_extra_layers': True
            },
            "clftnet": {
                'in_ch': 3, 'out_ch': 1, 'dim': 64, 'ori_h': 144, 'extra_fc': False, 'e_factor': [2, 4, 8, 16]
            },
            "clftfdia": {
                'in_ch': 3, 'out_ch': 1, 'dim': 64, 'ori_h': 144, 'extra_fc': False, 'e_factor': [2, 4, 8, 16],
                'ablated': [True, True], 'visualize': False
            },
            "demilensnetdev": {
                'in_ch': 3, 'out_ch': 1, 'dim': 64, 'ori_h': 144, 'extra_fc': False, 'e_factor': [2, 4, 8, 16],
                'ablated': [True, True], 'visualize': False
            },
            "demilensnet": {
                'in_ch': 3, 'out_ch': 1, 'dim': 64, 'ori_h': 144, 'extra_fc': False, 'e_factor': [2, 4, 8, 16],
                'ablated': [True, True], 'visualize': False
            },
            "demilensnet_ablation": {
                'in_ch': 3, 'out_ch': 1, 'dim': 64, 'ori_h': 144, 'extra_fc': False, 'e_factor': [2, 4, 8, 16],
                'ablated': [True, True], 'visualize': False
            },
            "clftnetcasa": {
                'in_ch': 3, 'out_ch': 1, 'dim': 64, 'ori_h': 144, 'extra_fc': False, 'e_factor': [2, 4, 8, 16]
            },
            "clftswintransformer": {
                'img_size': 144, 'patch_size': 4, 'in_chans': 3, 'num_classes': 1, 'embed_dim': 96,
                'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24], 'window_size': 6, 'mlp_ratio': 4,
                'drop_rate': 0., 'drop_path_rate': 0., 'ape': False, 'patch_norm': True
            },
            "cswin": {
            }
        }

        @classmethod
        def ensure_default_config(cls):
            """Ensure that default.yaml exists and is not modified"""
            default_config_path = Path(cls.DEFAULT_CONFIG)

            def _is_default_same():
                """Check if the default.yaml is the same as the backup"""
                if not default_config_path.exists():
                    return False, -1

                with default_config_path.open("r", encoding="utf-8") as f:
                    try:
                        loaded_default = yaml.safe_load(f)
                    except yaml.YAMLError:
                        return False, 0

                if loaded_default == cls.DEFAULT_CONFIG_BACKUP:
                    return True, 1
                else:
                    return False, 0

            flag, status = _is_default_same()
            if not flag:
                if status != 1:
                    if status == -1:
                        print(f"[\033[93mWarning\033[0m] Default config is missing. Restoring...")
                    elif status == 0:
                        print(f"[\033[93mWarning\033[0m] Default config is changed. Restoring...")
                    ModelSettings.save_config(external_save=True, config=cls.DEFAULT_CONFIG_BACKUP,
                                              path=str(default_config_path))

        @classmethod
        def generate_config(cls):
            """Prompt user for a config name and model name, then create a new config file from default.yaml with appropriate model_params."""
            new_config_path = None
            default_config_path = Path(cls.DEFAULT_CONFIG)
            config_dir = Path(cls.CONFIG_DIR)

            while True:
                config_name = input("Enter \033[94mconfig name\033[0m (\033[96mwithout .yaml\033[0m): ").strip()
                if not config_name:
                    print("[\033[91mError\033[0m] Config name cannot be empty.")
                    continue
                else:
                    if re.search(r'[<>:"|?*\x00-\x1F]', config_name):
                        print("\033[91m[Error]\033[0m Invalid filename.")
                        continue
                    config_name = f"{config_name}.yaml"
                    if input(f"Is \033[94m{config_name}\033[0m the name you want? (y/n): ").strip().lower() in [
                        'y', 'yes', '']:
                        new_config_path = config_dir / f"{config_name}"
                        if new_config_path.exists():
                            overwrite = input(
                                f"[\033[93mWarning\033[0m] {config_name} already exists. Overwrite? (y/n):").strip().lower()
                            if overwrite != 'y':
                                print("Operation canceled.\n")
                                time.sleep(1)
                                continue
                        break
                    else:
                        time.sleep(1)
                        print()

            model_name = input("Enter \033[94mmodel name\033[0m: ").strip()
            if model_name in cls.MODEL_PARAMS_MAP:
                model_params = cls.MODEL_PARAMS_MAP[model_name]
                print(f"\033[94mFind model '{model_name}' in the preset list, using predefined parameters.\033[0m")
            else:
                print(
                    f"[\033[93mWarning\033[0m] Model '{model_name}' not found in preset list. You need to import the model and manually define the parameters in config file.\033[0m")
                model_params = {}

            with default_config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            config['model']['model_name'] = model_name
            config['model']['model_params'] = model_params

            ModelSettings.save_config(external_save=True, config=config, path=str(new_config_path))

            print(
                f"\033[92mSuccessfully created {config_name} in {config_dir}. You can now edit it and then run main.py(with -g option) to generate your training script.\033[0m")

        @classmethod
        def _compare_configs(cls, current_config, default_config, path="", exclude=['model']):
            """Compare two config dictionaries, return miss and extra keys."""
            miss = []
            extra = []

            for key in default_config:
                if key in exclude:
                    continue
                new_path = f"{path}.{key}" if path else key
                if key not in current_config:
                    miss.append(new_path)

                elif isinstance(default_config[key], dict) and isinstance(current_config.get(key), dict):
                    sub_missing, sub_extra = cls._compare_configs(current_config[key], default_config[key], new_path,
                                                                  exclude)
                    miss.extend(sub_missing)
                    extra.extend(sub_extra)

            for key in current_config:
                if key in exclude:
                    continue
                if key not in default_config:
                    new_path = f"{path}.{key}" if path else key
                    extra.append(new_path)

            return miss, extra

        @classmethod
        def repair_config(cls, current_config, default_config):
            """Repair the current config by adding miss params from default_config, excluding 'model'."""
            miss, extra = cls._compare_configs(current_config, default_config)

            for path in miss:
                keys = path.split('.')
                default_val = default_config
                for k in keys:
                    default_val = default_val[k]
                current = current_config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                if keys[-1] not in current:
                    current[keys[-1]] = default_val
            return miss, extra


def parse_args():
    """Parse command line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Config management script for deep learning training initialization.",
        usage="python config.py [OPTION] [CONFIG_PATH]"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-g", "--generate",
        action="store_true",
        help="Create a new config file by copying 'default.yaml' with custom model name."
    )
    group.add_argument(
        "-r", "--repair",
        action="store",
        metavar="CONFIG_PATH",
        help="Repair an existing config file to match the structure of 'default.yaml'. Requires a config file path."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Mapping.ConfigSetup.ensure_default_config()

    if args.generate:
        Mapping.ConfigSetup.generate_config()
    elif args.repair:
        config_path = args.repair
        if not os.path.exists(config_path):
            print(
                f"[\033[91mError\033[0m] Config file \033[91m{config_path}\033[0m does not exist. (Is the content included?)")
            sys.exit(1)

        with open(config_path, "r", encoding="utf-8") as f:
            current_config = yaml.safe_load(f)

        missing, extras = Mapping.ConfigSetup.repair_config(current_config, Mapping.ConfigSetup.DEFAULT_CONFIG_BACKUP)

        if missing:
            print("\033[94mAdded missing parameters:\033[0m")
            for m in missing:
                print(f"- \033[96m{m}\033[0m")
        if extras:
            print("\033[94mExtra parameters found (you may remove them):\033[0m")
            for e in extras:
                print(f"- \033[96m{e}\033[0m")
        if not missing and not extras:
            print("\033[92mNo repairs needed.\033[0m")
        ModelSettings.save_config(external_save=True, config=current_config, path=config_path)
        print(f"\033[92mSuccessfully repaired {config_path}.\033[0m")
    else:
        print(
            "[\033[93mWarning]\033[0m] Direct execution is not allowed. Use \033[93m`python config.py -g`\033[0m to generate a new config file or \033[93m`python config.py -r <config_path>`\033[0m to repair a config file.")
        print("\033[91mAborted.\033[0m")
