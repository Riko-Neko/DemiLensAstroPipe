# Astronomical FITS/PNG Classification Pipeline

A user-oriented, highly interactive PyTorch deep learning pipeline based on `YAML` configuration. With preset astronomy-optimized models, it can be adapted to `FITS`/`PNG` classification tasks on task-specific data. In this repository, gravitational lens binary classification with `DemiLensNet` is the concrete task adaptation.

- supports `FITS` and `PNG` inputs
- supports directory-based datasets and `CSV`-indexed datasets
- supports training, evaluation, plotting, and offline prediction
- supports interactive configuration creation and run setup
- supports astronomy-oriented preset model templates for single-band and multi-band tasks

## Configuration Layer: `config.py`

The configuration layer is implemented in [`config.py`](config.py). It manages the task schema, model templates, runtime config loading, and config repair.

Core pieces:

- [`config/default.yaml`](config/default.yaml): default schema
- `MODEL_PARAMS_MAP`: preset model parameter templates
- `LOSS_MAP`, `OPTIMIZER_MAP`, `SCHEDULER_MAP`: training component registries
- `ModelSettings`: runtime configuration manager

### Main commands

```bash
python config.py -g
python config.py -r config/your_config.yaml
```

`python config.py -g`

- creates a new task config interactively
- asks for a config name
- asks for a `model_name`
- injects preset `model_params` when the selected model exists in `MODEL_PARAMS_MAP`

`python config.py -r <CONFIG_PATH>`

- repairs an existing config against the default schema
- restores missing keys
- reports extra keys for manual cleanup

### `default.yaml` as the reference example

[`config/default.yaml`](config/default.yaml) defines the standard structure of a task config. It is organized into five blocks:

- `flags`: runtime update control
- `path`: dataset and output locations
- `data`: input channels, image size, augmentation, normalization
- `train`: optimizer, loss, scheduler, learning rate, batch size, epochs
- `model`: `model_name` and `model_params`

Practical reading of the schema:

- edit `path` to point to the target dataset and output directories
- edit `data` to match band count, input size, and preprocessing strategy
- edit `train` to match the optimization schedule
- set `model_name` and refine `model_params`

Example model block:

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

### Runtime configuration behavior

`ModelSettings` loads the active `YAML`, selects the device, stores the parsed config tree, and exposes the `path`, `data`, `train`, and `model` sub-configs.

During training, `setup_training()` can detect changes on disk and apply supported updates in three categories:

- normal parameter updates
- dataset rebuild updates
- model rebuild updates

## Main Workflow: `main.py`

The main workflow entry is implemented in [`main.py`](main.py). It turns a configured task into a training, testing, plotting, or prediction run.

### Main entry points

```bash
python main.py -g
python main.py -t
python main.py -p
python main.py -P
python main.py -P 0.50
python main.py -P 0.45 0.60
```

Roles:

- `-g`: interactively set the initial run behavior and generate a reusable execution script
- `-t`: test one or more trained configs
- `-p`: plot test results
- `-P`: run prediction with optional thresholds

### What `main.py` controls

In generation mode, `main.py` configures:

- device index
- device information verbosity
- dataloader checking
- model summary generation
- saved-weight loading
- unmatched-weight loading
- checkpoint loading strictness
- epoch logging
- dynamic training updates
- final testing
- result analysis

### Workflow stages

The runtime stages are assembled in [`workflow.py`](workflow.py):

1. `Init Config`
2. `Set Device`
3. `Data Preparation`
4. `Build Model`
5. `Build Train Pipeline`
6. `Train Model`
7. `Test Model`
8. `Analyze Results`
9. `Model Prediction`

Related core files:

- data loading and augmentation: [`augment.py`](augment.py)
- training and validation: [`train.py`](train.py)
- testing and ROC export: [`test.py`](test.py)
- prediction: [`pred.py`](pred.py)
- plotting: [`plot.py`](plot.py)
- builders, checkpoints, logging: [`utils.py`](utils.py)

### Input and output

Input modes:

- directory mode for positive and negative sample folders
- `CSV` mode for indexed sample lists

Supported formats:

- `FITS`
- `PNG`

Common outputs:

- model weights
- checkpoints
- epoch logs
- test logs
- probability and ROC CSV files
- plotted figures
- threshold-split prediction folders

### Recommended usage flow

1. create or repair a config with `config.py`
2. edit `path`, `data`, `train`, and `model`
3. use `main.py` to define run behavior
4. train the model
5. test, plot, or predict with the workflow entry points

## DemiLensNet and the Model Library

### DemiLensNet

The implementation is in [`model/DemiLensNet.py`](model/DemiLensNet.py).

`DemiLensNet` is the dedicated gravitational lens classification model in this repository. It combines convolutional feature extraction with attention-driven refinement for weak, morphology-sensitive astronomical signals across multiple spatial scales.

Key characteristics:

- convolutional local feature modeling
- multi-scale structure capture through dilation and hierarchical expansion
- attention-based deep feature refinement
- deformable attention-style spatial masking
- support for single-band and multi-band inputs

### Main parameters

The main `DemiLensNet` parameters exposed through `model_params` are:

- `in_ch`
- `out_ch`
- `dim`
- `ori_h`
- `extra_fc`
- `e_factor`
- `ablated`
- `visualize`

Typical adaptation rules:

- change `in_ch` for single-band or multi-band data
- keep `ori_h` aligned with `data.image_size`
- scale `dim` and `e_factor` for model capacity
- use `ablated` for controlled ablation variants

### Model library in `config.py`

Project-specific models:

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

TorchVision backbones:

- `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- `resnext50_32x4d`, `resnext101_32x8d`
- `wide_resnet50_2`, `wide_resnet101_2`
- `vgg16`, `vgg19`
- `densenet121`, `densenet161`, `densenet169`, `densenet201`
- `mobilenet_v2`, `mobilenetv2`
- `mnasnet0_5`, `mnasnet0_75`, `mnasnet1_0`, `mnasnet1_3`
- `shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x1_5`, `shufflenet_v2_x2_0`
- `squeezenet1_0`, `squeezenet1_1`
- `alexnet`, `googlenet`, `inception_v3`, `inceptionv3`

### Preset model parameter templates

`MODEL_PARAMS_MAP` currently provides templates for:

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

These templates provide the starting parameter block used by `python config.py -g`.

### Paper

- [DemiLensNet](https://example.com/demilensnet-paper)

### License

This project uses the MIT License.
