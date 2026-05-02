# Astronomical FITS/PNG Classification Pipeline

A user-oriented, highly interactive PyTorch deep learning pipeline based on `YAML` configuration. With preset astronomy-optimized models, it can be adapted to a wide range of `FITS`/`PNG` classification tasks for task-specific data. In this repository, gravitational lens binary classification with `DemiLensNet` is the concrete task adaptation built on top of the pipeline.

- supports `FITS` and `PNG` inputs
- supports directory-based datasets and `CSV`-indexed datasets
- supports training, evaluation, plotting, and offline prediction
- supports interactive configuration creation
- supports astronomy-oriented preset model templates for single-band and multi-band tasks

## Configuration Layer: `config.py`

The configuration layer is implemented in [`config.py`](config.py). Its purpose is to define, validate, repair, and interpret the task configuration before training or inference starts.

At this layer, the project treats configuration as an abstract schema rather than a fixed experiment file. The key pieces are:

- the default schema in [`config/default.yaml`](config/default.yaml)
- the preset model parameter library in `MODEL_PARAMS_MAP`
- the training component registries in `LOSS_MAP`, `OPTIMIZER_MAP`, and `SCHEDULER_MAP`
- the runtime configuration manager `ModelSettings`

### What `config.py` does

`config.py` is responsible for:

- ensuring that the default config schema exists and remains structurally valid
- generating a new task config interactively
- repairing an existing config when fields are missing or outdated
- loading the active config into `ModelSettings`
- parsing model-related strings into actual Python objects
- checking and applying supported config updates during training

### Main `config.py` commands

```bash
python config.py -g
python config.py -r config/your_config.yaml
```

`python config.py -g`

- creates a new config interactively
- asks for a config name
- asks for a `model_name`
- if the model name exists in the preset model library, automatically injects a matching `model_params` template

`python config.py -r <CONFIG_PATH>`

- repairs an existing config against the default schema
- restores missing keys from the default structure
- reports extra keys for manual review

### Using `default.yaml` as the reference example

[`config/default.yaml`](config/default.yaml) should be read as the reference schema of the pipeline. It is not primarily a ready-to-run experiment. Its job is to define the expected structure of a valid task config.

The file is organized into five sections:

- `flags`
- `path`
- `data`
- `train`
- `model`

#### `flags`

This block controls runtime update behavior.

- `model_rebuild`
- `dataset_rebuild`
- `params_update`

In normal usage, these fields are usually left unchanged unless runtime configuration update behavior is intentionally used.

#### `path`

This block defines where the pipeline reads data and writes outputs.

Typical fields include:

- training and test data locations
- positive and negative sample directories
- `CSV` index paths
- prediction output directory
- test output directory
- plot output directory
- weights directory
- checkpoint directory
- log directory

When adapting the project to a new dataset, this is usually the first block to edit.

#### `data`

This block defines how image tensors are built.

Typical fields include:

- `input_channels`
- `image_size`
- `augment_mode`
- `adaptation_mode`
- `channel_expansion_mode`
- `norm`
- `mean`
- `std`

This is where the pipeline is adapted to single-band versus multi-band data, to different image sizes, and to different augmentation strategies.

#### `train`

This block defines the optimization and training schedule.

Typical fields include:

- `optimizer`
- `loss_function`
- `scheduler`
- `learning_rate`
- `batch_size`
- `num_epochs`
- `weight_decay`
- `save_interval`
- `self_evaluate_interval`
- `process_bar`

This is where learning rate, optimizer choice, batch size, epoch count, and training behavior are controlled.

#### `model`

This block defines the model itself through:

- `model_name`
- `model_params`

This is the block that turns the default schema into a concrete trainable task.

### Example reading of `default.yaml`

A typical use pattern is:

1. Start from the default schema.
2. Choose a model through `model_name`.
3. Accept or edit the injected `model_params`.
4. fill in `path` for the target dataset.
5. adjust `data` for channel count, image size, and augmentation.
6. adjust `train` for optimization strategy.

A minimal model block can look like this:

```yaml
model:
  model_name: demilensnet
  model_params:
    in_ch: 3
    out_ch: 1
    dim: 32
    ori_h: 144
    extra_fc: true
    ablated: [false, false]
    e_factor: [2, 4, 8, 16]
    visualize: false
```

This should be understood only as an example of how the schema is used, not as the only recommended task definition.

### Runtime behavior of the configuration layer

`ModelSettings` is the central runtime configuration object. It:

- loads the active `YAML`
- selects the device
- stores the parsed configuration tree
- exposes the `path`, `data`, `train`, and `model` sub-configs

During training, `setup_training()` can compare the in-memory config with the version on disk and apply supported updates in three categories:

- normal parameter updates
- dataset rebuild updates
- model rebuild updates

This is one of the reasons the pipeline is more interactive than a typical static training script.

## Main Workflow: `main.py`

The main workflow entry is implemented in [`main.py`](main.py). Its role is to turn a task configuration into an executable training, testing, plotting, or prediction process.

### Main `main.py` entry points

```bash
python main.py -g
python main.py -t
python main.py -p
python main.py -P
python main.py -P 0.50
python main.py -P 0.45 0.60
```

Their roles are:

- `-g`: interactively define the initial run behavior and generate a reusable execution script
- `-t`: test one or more trained configs
- `-p`: plot test results
- `-P`: run prediction with optional threshold values

### What `main.py` controls

`main.py` does not define the model family or the dataset schema. Instead, it controls how a configured task is started.

In generation mode, it asks whether to:

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

This means:

- `config.py` defines the task
- `main.py` defines the execution behavior of that task

### Workflow stages

The actual runtime stages are assembled in [`workflow.py`](workflow.py):

1. `Init Config`
2. `Set Device`
3. `Data Preparation`
4. `Build Model`
5. `Build Train Pipeline`
6. `Train Model`
7. `Test Model`
8. `Analyze Results`
9. `Model Prediction`

These stages connect the core files of the repository:

- data loading and augmentation: [`augment.py`](augment.py)
- training and validation: [`train.py`](train.py)
- test-time metrics and ROC export: [`test.py`](test.py)
- threshold-based prediction export: [`pred.py`](pred.py)
- plotting and post-test visualization: [`plot.py`](plot.py)
- builders, checkpoints, and logging: [`utils.py`](utils.py)

### Input modes

The workflow supports two main input modes:

- directory mode, where positive and negative samples are provided as folders
- `CSV` mode, where the sample list and labels are defined in a table

Supported formats are:

- `FITS`
- `PNG`

The final tensor behavior is mainly controlled by:

- `data.input_channels`
- `data.image_size`
- `data.augment_mode`
- `data.adaptation_mode`
- `data.norm`

### Output artifacts

The workflow writes artifacts to the configured output directories, typically including:

- model weights
- checkpoints
- epoch logs
- test logs
- probability CSV files
- ROC CSV files
- plotted figures
- prediction output folders split by threshold result

### Recommended user flow

For a normal task, the recommended order is:

1. create or repair a config with `config.py`
2. edit the config so `path`, `data`, `train`, and `model` match the target task
3. use `main.py` to define the run behavior
4. train the model
5. test, plot, or predict through the provided workflow entry points

## DemiLensNet and the Model Library

### DemiLensNet

The implementation of the model is in [`model/DemiLensNet.py`](model/DemiLensNet.py).

`DemiLensNet` is the dedicated gravitational lens classification model in this repository. It is designed for astronomical image recognition scenarios where the target signal can be weak, morphologically subtle, and distributed across multiple spatial scales.

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

The model registry is defined in [`config.py`](config.py).

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

Besides the model registry, `config.py` also exposes a preset template library through `MODEL_PARAMS_MAP`.

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
