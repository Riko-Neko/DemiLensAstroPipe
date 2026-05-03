from typing import List, Tuple, Dict, Any, Callable, Optional

import torch

from augment import DatasetToolkit
from config import ModelSettings, Interface
from plot import plot_from_workflow, plot_from_config
from pred import predict
from test import test
from train import train, evaluate
from utils import BuilderManager, CheckpointManager, TrainingPipelineBuilder

# Shared instance
dataloader_train: torch.utils.data.DataLoader = None  # fixed train dataloader type
dataloader_validation: torch.utils.data.DataLoader = None  # fixed validation dataloader type
model: None  # allow self-defined model
checkpoint: Dict[str, Any] = None  # allow self-defined checkpoint (dict format)
criterion: torch.nn.Module = None  # fixed criterion type
optimizer: None  # allow self-defined optimizer
lr_scheduler: None  # internal defined lr_scheduler
epoch_logger: Optional[Callable] = None  # allow logging for per epoch


def init_config(config_name: str, use_default: bool = False):
    # print(f"Initializing config: {config_name} (use_default={use_default})")
    ModelSettings.init_config(config_name, use_default=use_default)


def set_device(device_index: int = 0, verbose: bool = True, info_verbose: bool = True):
    # print(f"Setting device: index={device_index}, info_verbose={info_verbose}")
    ModelSettings.set_device(device_index=device_index, verbose=verbose, info_verbose=info_verbose)
    torch.cuda.empty_cache()


def data_preparation(apply_loader_check: bool = True, check_args: Tuple[Any] = (3, 0),
                     check_kwargs: Dict[str, Any] = {"verbose": False, "calculate_proportion": True}):
    global dataloader_train, dataloader_validation
    # print("Preparing data...")
    builder_manager = BuilderManager()
    dataloader_train, dataloader_validation = builder_manager.data_builder()

    if apply_loader_check:
        print(f"Checking loaders with args={check_args}, kwargs={check_kwargs}")
        DatasetToolkit.check_loader(dataloader_train, *check_args, **check_kwargs)
        DatasetToolkit.check_loader(dataloader_validation, *check_args, **check_kwargs)


def build_model(generate_summary: bool = True):
    global model
    # print(f"Building model (generate_summary={generate_summary})")
    builder_manager = BuilderManager()
    model = builder_manager.model_builder(generate_summary=generate_summary)


def build_train_pipeline(revise_accuracy: bool = True, log_epochs: bool = True,
                         load_weights_kwargs: Dict[str, Any] = {"unmatch": True},
                         eval_kwargs: Dict[str, Any] = {"dynamic_auc": True},
                         load_checkpoint_kwargs={"use_saved_weights": False, "strict_load": True}):
    global optimizer, criterion, lr_scheduler, checkpoint, epoch_logger, test_logger
    # print("Building train pipeline...")
    checkpoint_manager = CheckpointManager()
    training_pipeline = TrainingPipelineBuilder()

    criterion = training_pipeline.criterion_builder()
    optimizer = training_pipeline.optimizer_builder(model)
    lr_scheduler = training_pipeline.lr_scheduler_builder(optimizer)
    if log_epochs:
        training_pipeline.setup_logging()
        epoch_logger = training_pipeline.log_epoch

    if revise_accuracy or load_checkpoint_kwargs["use_saved_weights"]:
        checkpoint_manager.load_weights(model, **load_weights_kwargs)
        eval_value = evaluate(model, dataloader_validation, criterion, **eval_kwargs) if revise_accuracy else None
    else:
        eval_value = None

    checkpoint = checkpoint_manager.load_checkpoint(model, optimizer, revised_value=eval_value,
                                                    **load_checkpoint_kwargs)


def train_model(dynamic_update: bool = False):
    # print(f"Training model (dynamic_update={dynamic_update})")
    train(model, [dataloader_train, dataloader_validation], criterion, optimizer, lr_scheduler, checkpoint=checkpoint,
          dynamic_update=dynamic_update, log_func=epoch_logger)

    CheckpointManager().end_save(model, optimizer, checkpoint)
    torch.cuda.empty_cache()


def test_model(test_alone: bool = False, device_index: int = 0):
    # print("Testing model (multi_test={multi_test})")
    if not test_alone:
        test(model, dataloader_validation, criterion)
    else:
        paths = ModelSettings.user_config_lib
        hide_process = len(paths) > 1
        set_device(device_index, info_verbose=False)
        torch.cuda.empty_cache()

        for path in paths:
            print()
            print(f"\033[94mPreparing model test from \033[96m{path}\033[94m\033[0m")
            ModelSettings.init_config(path, full_path=True)
            builder_manager = BuilderManager()
            print("\033[94mPreparing test data...\033[0m")
            dataloader_test = builder_manager.data_builder(test_mode=True, hide_process=hide_process)
            print(f"\033[92mDone!\033[0m ==> \033[96m{len(dataloader_test.dataset)} test samples\033[0m")
            print(f"\033[94mBuilding model...\033[0m", end='')
            t_model = builder_manager.model_builder(generate_summary=False)
            print(f"\033[92mDone!\033[0m ==> \033[96m{t_model.__class__.__name__}\033[0m")
            checkpoint_manager = CheckpointManager()
            checkpoint_manager.load_weights(t_model, unmatch=False, verbose=False)
            training_pipeline = TrainingPipelineBuilder()
            t_criterion = training_pipeline.criterion_builder(verbose=False)
            print(f"\033[94mStarting Test...\033[0m\nWeights Loaded\n(Loss function: {t_criterion.__class__.__name__})")
            test(t_model, dataloader_test, t_criterion, log_func=training_pipeline.log_test_result)

        print("\033[92mAll tests done!\033[0m")


def analyze_results(plot_alone: bool = False):
    # print("Analyzing results...")
    if not plot_alone:
        plot_from_workflow()
    else:
        plot_from_config()


def predict_model(device_index: int = 0, thresholds: list = None):
    # print(f"Starting prediction...")
    paths = ModelSettings.user_config_lib
    set_device(device_index, info_verbose=False)
    torch.cuda.empty_cache()
    dirs = []

    label = input(
        "Enter the prior label distribution (\033[96m0\033[0m or \033[96m1\033[0m, or \033[96mEnter\033[0m to skip): ")
    if label == '0':
        label_instr = 'all_0'
    elif label == '1':
        label_instr = 'all_1'
    else:
        label_instr = None

    if thresholds is None:
        print(f"[\033[92mInfo\033[0m] Do you want to update thresholds?")
        print(
            f"Enter \033[96m{len(paths)}\033[0m space-separated thresholds (0–1), Press Enter directly to use logged/default values.")

        user_input = input("Thresholds: ").strip()

        if user_input == "":
            thresholds = [None] * len(paths)
            print("[\033[92mInfo\033[0m] Using logged/default values.")
        else:
            try:
                # split input string by spaces and convert to float
                thresholds = [float(x) for x in user_input.split()]
                # check range validity
                for i, v in enumerate(thresholds):
                    if not (0 <= v <= 1):
                        print(f"[\033[91mWarning\033[0m] Value {v} out of range [0,1], using logged/default values.")
                        thresholds[i] = None
                # check length match
                if len(thresholds) != len(paths):
                    print(
                        f"[\033[91mWarning\033[0m] Provided {len(thresholds)} thresholds, expected {len(paths)}. Missing values will be filled with logged/default values.")
                    thresholds += [None] * (len(paths) - len(thresholds))
                print(f"[\033[96mInfo\033[0m] Final thresholds: {thresholds}")
            except ValueError:
                print("[\033[91mError\033[0m] Invalid input. Using logged/default values.")
                thresholds = [None] * len(paths)

    elif len(thresholds) != len(paths):
        print(
            "[\033[91mERROR\033[0m] The thresholds provided and the configs provided are not matched, using logged values.")
        thresholds = [None] * len(paths)

    for path, threshold in zip(paths, thresholds):
        print()
        print(f"\033[94mPreparing prediction from \033[96m{path}\033[94m\033[0m")
        ModelSettings.init_config(path, full_path=True)
        builder_manager = BuilderManager()
        dataloader_pred = builder_manager.data_builder(pred_mode=True, hide_process=True)
        print(f"\033[94mBuilding model\033[0m", end='')
        p_model = builder_manager.model_builder(generate_summary=False)
        print(f" ==> \033[96m{p_model.__class__.__name__}\033[0m")
        checkpoint_manager = CheckpointManager()
        checkpoint_manager.load_weights(p_model, unmatch=False, verbose=True)
        training_pipeline = TrainingPipelineBuilder()
        test_log = training_pipeline.read_test_log()
        threshold = test_log["best_threshold"] if threshold is None else threshold
        if threshold is None:
            threshold = 0.5
            print(
                f"[\033[93mWarning\033[0m] No threshold found(Using default). You may forget to test the model. It`s recommended to execute model test first to maximize performance.")
        print(f"Threshold: \033[96m{threshold:.2f}\033[0m")
        output_dir, precision, recall, FNR = predict(p_model, dataloader_pred, threshold=threshold,
                                                     label_instr=label_instr)
        if label_instr is not None:
            print(f"\n\033[92m{label_instr} | Recall: {recall:.4f}, Precision: {precision:.4f}, FNR: {FNR:.4f}\033[0m")
        dirs.append(output_dir)

    print("\033[92mAll Predictions done!\033[0m")
    print("\033[92mCheck results in:\033[0m")
    for dir in dirs:
        print(f"{dir}")


def create_workflow(
        config_name: str = "default.yaml",
        use_default: bool = False,
        device_index: int = 0,
        device_info_verbose: bool = True,
        apply_loader_check: bool = True,
        data_check_args: Tuple[Any] = (3, 0),
        data_check_kwargs: Dict[str, Any] = {"verbose": False, "calculate_proportion": True},
        model_generate_summary: bool = True,
        revise_accuracy: bool = True,
        use_saved_weights: bool = False,
        weights_unmatch_load: bool = True,
        checkpoint_strict_load: bool = True,
        log_epochs: bool = True,
        train_dynamic_update: bool = False,
        final_test: bool = True,
        results_analysis: bool = True,
        test_only: bool = False,
        plot_only: bool = False,
        predict: bool = False,
        thresholds: float = None,

) -> List["Interface.WorkflowExecutor"]:
    func_list = {
        1: Interface.WorkflowExecutor(
            name="Init Config",
            function=init_config,
            kwargs={"config_name": config_name, "use_default": use_default}
        ),
        2: Interface.WorkflowExecutor(
            name="Set Device",
            function=set_device,
            kwargs={"device_index": device_index, "verbose": True, "info_verbose": device_info_verbose}
        ),
        3: Interface.WorkflowExecutor(
            name="Data Preparation",
            function=data_preparation,
            kwargs={"apply_loader_check": apply_loader_check, "check_args": data_check_args,
                    "check_kwargs": data_check_kwargs}
        ),
        4: Interface.WorkflowExecutor(
            name="Build Model",
            function=build_model,
            kwargs={"generate_summary": model_generate_summary}
        ),
        5: Interface.WorkflowExecutor(
            name="Build Train Pipeline",
            function=build_train_pipeline,
            kwargs={"revise_accuracy": revise_accuracy, "log_epochs": log_epochs,
                    "load_weights_kwargs": {"unmatch": weights_unmatch_load}, "eval_kwargs": {"dynamic_auc": True},
                    "load_checkpoint_kwargs": {"use_saved_weights": use_saved_weights,
                                               "strict_load": checkpoint_strict_load}}
        ),
        6: Interface.WorkflowExecutor(
            name="Train Model",
            function=train_model,
            kwargs={"dynamic_update": train_dynamic_update}
        ),
        7: Interface.WorkflowExecutor(
            name="Test Model",
            function=test_model,
            kwargs={"test_alone": test_only, "device_index": device_index}
        ),
        8: Interface.WorkflowExecutor(
            name="Analyze Results",
            function=analyze_results,
            kwargs={"plot_alone": plot_only}
        ),
        9: Interface.WorkflowExecutor(
            name="Model Prediction",
            function=predict_model,
            kwargs={"device_index": device_index, "thresholds": thresholds}
        )
    }
    if predict:
        exe_list = [9]
    elif test_only and plot_only:
        exe_list = [7, 8]
    elif plot_only:
        exe_list = [8]
    elif test_only:
        exe_list = [7]
    elif final_test:
        exe_list = [1, 2, 3, 4, 5, 6, 7]
    elif results_analysis:
        exe_list = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        exe_list = [1, 2, 3, 4, 5, 6]
    workflow = [func_list[i] for i in exe_list if i in func_list]
    return workflow
