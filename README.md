# Astronomical FITS/PNG Classification Pipeline

A user-oriented, highly interactive PyTorch deep learning pipeline based on `YAML` configuration. With preset astronomy-optimized models, it can be adapted to a wide range of `FITS`/`PNG` classification tasks for task-specific data. In this repository, gravitational lens binary classification with `DemiLensNet` is the concrete task adaptation built on top of the pipeline.

The pipeline supports:

- `FITS` and `PNG` inputs
- directory-based datasets and `CSV`-indexed datasets
- training, evaluation, plotting, and offline prediction
- interactive configuration creation
- interactive training script generation
- astronomy-oriented model presets that can be adapted to single-band or multi-band classification tasks

## Configuration and Workflow

### The Three-Layer Entry Structure

The project is easiest to use if you read it as three layers:

1. `config.py`: configuration initialization, update, and repair
2. `main.py`: training script generation and initial run behavior
3. user-generated execution scripts: run-level parameter wrappers created from `main.py`

This is the intended usage model of the project. The first two layers are the pipeline itself; the third layer is a user-produced execution wrapper derived from those layers.

### Layer 1: `config.py`

File: [config.py](/Users/rikoneko/Documents/Projects/KiDS Lens Search/MAIN/config.py)

`config.py` manages the configuration layer itself. It is responsible for creating, validating, repairing, and interpreting `YAML` configs before a training or inference run begins.

Its role can be summarized as:

- ensure that `default.yaml` exists and remains structurally valid
- generate a new config interactively
- repair an existing config so it matches the expected schema
- load runtime configuration into `ModelSettings`
- parse model-related objects from strings into actual Python classes
- detect and apply certain config updates during training

#### What “preset configuration” means in this project

In the first part of this README, “preset configuration” should be understood as an abstract configuration mechanism, not as a specific experiment file under `config/`.

The preset layer consists of:

- a structural template: [config/default.yaml](/Users/rikoneko/Documents/Projects/KiDS Lens Search/MAIN/config/default.yaml)
- built-in model parameter templates in `config.py` through `MODEL_PARAMS_MAP`
- built-in training component registries such as `LOSS_MAP`, `OPTIMIZER_MAP`, and `SCHEDULER_MAP`

In other words, the project does not require users to start from a fixed experiment YAML. Instead, it expects them to start from a default schema and then specialize that schema through model templates and task-specific values.

#### `config.py` commands

```bash
python config.py -g
python config.py -r config/your_config.yaml
```

`python config.py -g` creates a new config interactively. You provide a config name and a `model_name`; if that model exists in the built-in model template library, the script injects a matching `model_params` block automatically.

`python config.py -r <CONFIG_PATH>` repairs an existing config. This is useful when the config structure is outdated or incomplete. Missing keys are restored from the default schema, while extra keys are reported for manual review.

#### `default.yaml` as the reference schema

File: [config/default.yaml](/Users/rikoneko/Documents/Projects/KiDS Lens Search/MAIN/config/default.yaml)

`default.yaml` is best treated as the canonical schema of the pipeline. It is not primarily a ready-to-run experiment file. Its job is to define the expected structure and default behavior of the configuration tree.

The file is organized into five sections:

- `flags`: runtime update flags such as `model_rebuild`, `dataset_rebuild`, and `params_update`
- `path`: data sources and output locations
- `data`: input shape, augmentation, normalization, and loading behavior
- `train`: optimizer, loss, scheduler, learning rate, batch size, and logging behavior
- `model`: `model_name` and `model_params`

#### How to read `default.yaml`

`flags`

- Controls whether certain classes of changes can be applied during training.
- In normal usage, users usually leave these alone unless they explicitly use runtime config updates.

`path`

- Defines where the training, testing, and prediction data live.
- Defines where weights, checkpoints, logs, CSV outputs, figures, and predictions are written.
- This is usually the first block that must be edited when adapting the project to a new dataset.

`data`

- Defines the input channel count, image size, augmentation mode, normalization behavior, and related preprocessing choices.
- This block is where users adapt the pipeline to single-band versus multi-band data, and to different image sizes.

`train`

- Defines the optimization strategy and training schedule.
- This includes the optimizer, loss function, scheduler, learning rate, batch size, epoch count, checkpoint frequency, and progress bar behavior.

`model`

- Defines the model entry point through `model_name`.
- Defines the model constructor arguments through `model_params`.
- This is the final block that turns the default schema into a trainable task definition.

#### Runtime configuration behavior

`ModelSettings` is the central runtime configuration object. It loads the active `YAML`, selects the device, stores the parsed configuration tree, and exposes sub-configs such as `path`, `data`, `train`, and `model`.

During training, `setup_training()` can compare the in-memory config with the version on disk and apply updates according to the project’s own rules:

- normal parameter updates
- dataset rebuild updates
- model rebuild updates

This makes the project more interactive than a typical static training script.

### Layer 2: `main.py`

File: [main.py](/Users/rikoneko/Documents/Projects/KiDS Lens Search/MAIN/main.py)

`main.py` is not the model definition layer and not the dataset definition layer. Its job is to transform a configuration into a concrete run plan and, when requested, produce a user-side execution wrapper for repeated runs.

It provides the following user-facing entry points:

```bash
python main.py -g
python main.py -t
python main.py -p
python main.py -P
python main.py -P 0.50
python main.py -P 0.45 0.60
```

Their roles are:

- `-g`: generate a dedicated execution script interactively
- `-t`: test one or more trained configs
- `-p`: plot test results
- `-P`: run prediction with optional thresholds

When used in generation mode, `main.py` asks for the initial execution behavior of the workflow. This includes whether to:

- choose a device index
- print detailed device information
- check dataloaders before training
- generate a model summary
- revise accuracy before continuing
- load saved weights
- allow unmatched weight loading
- load checkpoints strictly
- record per-epoch logs
- enable dynamic training updates
- run final testing
- run result analysis

This layer defines how the workflow starts, not what the task fundamentally is.

### Layer 3: User-Generated Execution Script Layer

The third layer is the execution script produced by `python main.py -g`. This script is generated from user choices and belongs to the usage layer of the project, not to the core pipeline definition.

Its purpose is to freeze the chosen startup behavior into a reusable run entry point. This makes repeated experiments simpler and more reproducible without changing the underlying pipeline code.

This script typically exposes only a small runtime surface:

- `-l` / `--load`: force `revise_accuracy=True` and `use_saved_weights=True`
- `-d` / `--device`: override the default device index stored in the script

This is the third layer of the system:

- `config.py` defines the task structure
- `main.py` defines the initial execution choices
- the generated execution script applies those choices repeatedly at run time

### Main Workflow Behavior

File: [workflow.py](/Users/rikoneko/Documents/Projects/KiDS Lens Search/MAIN/workflow.py)

The project’s runtime behavior is assembled as a workflow of named stages:

1. `Init Config`
2. `Set Device`
3. `Data Preparation`
4. `Build Model`
5. `Build Train Pipeline`
6. `Train Model`
7. `Test Model`
8. `Analyze Results`
9. `Model Prediction`

These stages map to the core code files:

- data loading and augmentation: [augment.py](/Users/rikoneko/Documents/Projects/KiDS Lens Search/MAIN/augment.py)
- training and evaluation: [train.py](/Users/rikoneko/Documents/Projects/KiDS Lens Search/MAIN/train.py)
- test-time statistics and ROC export: [test.py](/Users/rikoneko/Documents/Projects/KiDS Lens Search/MAIN/test.py)
- prediction and threshold-based file export: [pred.py](/Users/rikoneko/Documents/Projects/KiDS Lens Search/MAIN/pred.py)
- plotting and post-test visualization: [plot.py](/Users/rikoneko/Documents/Projects/KiDS Lens Search/MAIN/plot.py)
- builders, checkpoints, and logging: [utils.py](/Users/rikoneko/Documents/Projects/KiDS Lens Search/MAIN/utils.py)

### Input and Output Conventions

#### Inputs

The data layer supports two common modes:

- directory mode, where positive and negative samples are provided as folders
- `CSV` mode, where the sample list and labels are defined in a table

Supported image formats are:

- `FITS`
- `PNG`

The final tensor shape and preprocessing behavior are controlled by the config, especially:

- `data.input_channels`
- `data.image_size`
- `data.augment_mode`
- `data.adaptation_mode`
- `data.norm`

#### Outputs

The workflow writes artifacts to the configured output directories, typically including:

- model weights
- checkpoints
- epoch logs
- test logs
- probability CSV files
- ROC CSV files
- plotted figures
- prediction output folders split by threshold outcome

### Basic User Flow

For a normal task, the expected flow is:

1. Create or repair a config with `config.py`
2. Edit the config so `path`, `data`, `train`, and `model` match the task
3. Generate a dedicated training script with `main.py -g`
4. Train with the user-generated execution script
5. Test, plot, or predict with `main.py`

This is the intended user-facing path through the pipeline.

### Environment Notes

The repository does not currently provide a single dependency manifest in the root. Based on the imports used by the codebase, a working environment typically needs:

- `torch`
- `torchvision`
- `torchmetrics`
- `torchinfo`
- `pyyaml`
- `pandas`
- `matplotlib`
- `Pillow`
- `astropy`
- `scipy`
- `einops`
- `tqdm`
- `tabulate`
- `lion-pytorch`

For GPU usage, a valid CUDA environment and `nvidia-smi` are recommended.

## DemiLensNet and the Model Library

### DemiLensNet

File: [model/DemiLensNet.py](/Users/rikoneko/Documents/Projects/KiDS Lens Search/MAIN/model/DemiLensNet.py)

`DemiLensNet` is the project’s dedicated gravitational lens classification model. It is designed for astronomical image recognition scenarios where the target signal can be weak, morphologically subtle, and distributed across multiple spatial scales.

At a high level, the model combines convolutional feature extraction with attention-driven feature refinement. In practical terms, its design emphasizes:

- stable local texture modeling through convolutional blocks
- multi-scale structure capture through dilation and hierarchical feature expansion
- attention-based refinement in deeper feature stages
- deformable attention-style spatial masking for morphology-sensitive regions
- compatibility with both single-band and multi-band inputs

The model is intended as a task-oriented astronomical classifier rather than a generic off-the-shelf vision backbone.

### DemiLensNet Parameters

The main `DemiLensNet`-family parameters exposed through `model_params` are:

- `in_ch`: input channel count
- `out_ch`: output channel count, typically `1` for binary classification
- `dim`: base feature width
- `ori_h`: reference spatial size used by the model
- `extra_fc`: whether to use the extra fully connected head
- `e_factor`: hierarchical feature expansion factors
- `ablated`: switches for ablation variants
- `visualize`: whether to export internal visualization outputs

In practice:

- change `in_ch` when moving between single-band and multi-band data
- keep `ori_h` consistent with `data.image_size`
- scale `dim` and `e_factor` when changing model capacity
- use `ablated` only when intentionally running architecture ablations

### Model Library Exposed by `config.py`

The model registry is defined in [config.py](/Users/rikoneko/Documents/Projects/KiDS Lens Search/MAIN/config.py).

#### Project-specific models

- `demilensnet`
- `demilensnetdev`
- `demilensnet_ablated`
- `clftfdia`
- `clftnet`
- `clftnetcasa`
- `clftswintransformer`
- `demilensesswin`
- `quantumswin`
- `cswin`
- `vit`
- `resnet_specified`
- `swintransformerv2`
- `swin_mlp`
- `swin_transformer_moe`

#### TorchVision backbones

- `resnet18`
- `resnet34`
- `resnet50`
- `resnet101`
- `resnet152`
- `resnext50_32x4d`
- `resnext101_32x8d`
- `wide_resnet50_2`
- `wide_resnet101_2`
- `vgg16`
- `vgg19`
- `densenet121`
- `densenet161`
- `densenet169`
- `densenet201`
- `mobilenet_v2`
- `mobilenetv2`
- `mnasnet0_5`
- `mnasnet0_75`
- `mnasnet1_0`
- `mnasnet1_3`
- `shufflenet_v2_x0_5`
- `shufflenet_v2_x1_0`
- `shufflenet_v2_x1_5`
- `shufflenet_v2_x2_0`
- `squeezenet1_0`
- `squeezenet1_1`
- `alexnet`
- `googlenet`
- `inception_v3`
- `inceptionv3`

### Preset Model Parameter Templates

Besides the model registry, `config.py` also exposes a preset template library through `MODEL_PARAMS_MAP`. This is the abstract “preset configuration” layer for model construction.

Templates are currently provided for:

- `vit`
- `swintransformerv2`
- `resnet_specified`
- `clftnet`
- `clftfdia`
- `demilensnetdev`
- `demilensnet`
- `demilensnet_ablation`
- `clftnetcasa`
- `clftswintransformer`
- `cswin`

These templates do not replace task-specific configuration. They provide a model-shaped starting point that is injected into a new config when `python config.py -g` is used.

### Common Parameter Patterns Across the Model Library

Although different models expose different constructor signatures, the library broadly follows a few parameter families:

- convolutional astronomy models such as `demilensnet` and `clftnet` usually use `in_ch`, `out_ch`, `dim`, `ori_h`, `extra_fc`, and `e_factor`
- transformer-style models usually use `img_size`, `patch_size`, `embed_dim`, `depths`, `num_heads`, `window_size`, and related drop-rate settings
- custom ResNet-based entries usually expose `in_chans` and architecture-specific flags such as `use_extra_layers`

This means that, in most cases, adapting a model to a new astronomical classification task is mainly about keeping the channel count, image size, and model family parameters consistent with the data definition in the config.

### Paper Placeholder

The paper link for `DemiLensNet` is intentionally left as a placeholder until a public manuscript or project page is available:

- [DemiLensNet Paper Placeholder](https://example.com/demilensnet-paper)

### MIT License Statement

This project is intended to follow the MIT License. A separate root `LICENSE` file is not created here, but the code already includes MIT-style licensing language.

In practical terms, the MIT license allows users to use, copy, modify, merge, publish, distribute, sublicense, and sell the software, provided that the copyright notice and permission notice are retained. The software is provided as-is, without warranty.
