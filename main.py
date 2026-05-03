import argparse
import re
import sys

from workflow import create_workflow

__version__ = "1.0.1"


def print_info():
    print("""\
Copyright (c) 2023 LI

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:  

The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.  

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.  
""")


def print_args():
    print("""  
Project Information:  
This project allows users to generate custom-configured scripts based on their needs for creating and running workflows. Below are the meanings of each configuration variable:  

- config_name: The name of the configuration file, used to specify the configuration file for the workflow.  
- device_index: The device index, used to specify the index of the hardware device (such as GPU) being used.  
- device_info_verbose: Whether to enable detailed device information output; selecting 'y' will provide more detailed device information.  
- apply_loader_check: Whether to apply a data loader check; selecting 'y' will check the data loader before running.  
- data_check_args: Parameters for data checking, including max_batch and start_batch, used to specify the batch range for data checking.  
- data_check_kwargs: Keyword arguments for data checking, including whether to enable detailed checking output and whether to calculate ratios.  
- model_generate_summary: Whether to generate a model summary; selecting 'y' will generate summary information about the model before running.  
- revise_accuracy: Whether to revise accuracy; selecting 'y' will revise accuracy during runtime.  
- use_saved_weights: Whether to use saved weights; selecting 'y' will load and use previously saved model weights.  
- weights_unmatch_load: Whether to allow loading mismatched weights; selecting 'y' will allow loading in the case of weight mismatches.  
- checkpoints_strict_load: Whether to enable strict checkpoint loading; selecting 'y' will restrict loading checkpoints to those that fully match the model.  
- log_epochs: Whether to log accuracy and loss per epoch; selecting 'y' will apply logging during training.  
- train_dynamic_update: Whether to enable dynamic updates during training; selecting 'y' will enable a dynamic update mechanism during the training process.  
- test_and_analyze: Whether to enable final test and analysis; selecting 'y' will enable the final test and analysis process.  
""")


def prompt_for_config():
    config_name = input("Enter \033[94mconfig name\033[0m (\033[96mdefault: default.yaml\033[0m): ").strip()
    if not config_name:
        use_default = input(
            " --Use \033[94mdefault settings\033[0m? (y/n, \033[96mdefault: y\033[0m): ").strip().lower()
        if use_default == 'n':
            return prompt_for_config()
        config_name = "default.yaml"

    if re.search(r'[<>:"/\\|?*\x00-\x1F]', config_name):
        print("[\033[91mError\033[0m] Invalid filename.")
        return prompt_for_config()

    if not config_name.lower().endswith(('.yaml', '.yml')):
        config_name = re.sub(r'\.[^.]+$', '', config_name) + ".yaml"

    device_index = int(input("Enter \033[94mdevice index\033[0m (\033[96mdefault: 0\033[0m): ") or 0)
    device_info_verbose = input(
        "Enable \033[94mdevice info verbose\033[0m? (y/n, \033[96mdefault: y\033[0m): ").strip().lower() != 'n'
    apply_loader_check = input(
        "Apply \033[94mloader check\033[0m? (y/n, \033[96mdefault: y\033[0m): ").strip().lower() != 'n'

    config = {
        "config_name": config_name,
        "device_index": device_index,
        "device_info_verbose": device_info_verbose,
        "apply_loader_check": apply_loader_check,
    }

    if apply_loader_check:
        config.update({
            "data_check_args": (
                int(input(" --Enter \033[94mmax batches\033[0m for data check (\033[96mdefault: 3\033[0m): ") or 3),
                int(input(" --Enter \033[94mstart batch\033[0m for data check (\033[96mdefault: 0\033[0m): ") or 0),
            ),
            "data_check_kwargs": {
                "verbose": input(
                    " --Enable \033[94mverbose data check\033[0m? (y/n, \033[96mdefault: n\033[0m): "
                ).strip().lower() == 'y',
                "calculate_proportion": input(
                    " --Calculate \033[94mproportion\033[0m? (y/n, \033[96mdefault: y\033[0m): "
                ).strip().lower() != 'n',
            }
        })

    config.update({
        "model_generate_summary": input(
            "Generate \033[94mmodel summary\033[0m? (y/n, \033[96mdefault: y\033[0m): "
        ).strip().lower() != 'n',
        "revise_accuracy": input(
            "Apply \033[94maccuracy revising\033[0m? (y/n, \033[96mdefault: y\033[0m): "
        ).strip().lower() != 'n',
        "use_saved_weights": input(
            "Use \033[94msaved weights\033[0m? (y/n, \033[96mdefault: n\033[0m): "
        ).strip().lower() == 'y',
    })

    if config["revise_accuracy"] or config["use_saved_weights"]:
        config.update({
            "weights_unmatch_load": input(
                " --Allow \033[94munmatched weights loading\033[0m? (y/n, \033[96mdefault: y\033[0m): "
            ).strip().lower() != 'n'
        })

    config.update({
        "checkpoint_strict_load": input(
            "Enable \033[94mstrict checkpoint loading\033[0m? (y/n, \033[96mdefault: y\033[0m): "
        ).strip().lower() != 'n',
        "log_epochs": input(
            "Enable \033[94mepoch logging\033[0m? (y/n, \033[96mdefault: y\033[0m): "
        ).strip().lower() != 'n',
        "train_dynamic_update": input(
            "Enable \033[94mtrain dynamic update\033[0m? (y/n, \033[96mdefault: n\033[0m): "
        ).strip().lower() == 'y',
        "final_test": input(
            "Enable \033[94mfinal test\033[0m? (y/n, \033[96mdefault: y\033[0m): "
        ).strip().lower() != 'n',
        "results_analysis": input(
            "Enable \033[94mresults analysis\033[0m? (y/n, \033[96mdefault: y\033[0m): "
        ).strip().lower() != 'n'
    })

    if config["results_analysis"] and not config["final_test"]:
        print(
            "[\033[93mWarning\033[0m] Results analysis requires data from \033[93mfinal test\033[0m. Enabling \033[93mfinal\033[0m test automatically.")
        config["final_test"] = True

    return config


def generate_main_script(config):
    default_entries = [
        f'    "config_name": "{config["config_name"]}",',
        f'    "device_index": {config["device_index"]},',
        f'    "device_info_verbose": {config["device_info_verbose"]},',
        f'    "apply_loader_check": {config["apply_loader_check"]},'
    ]

    if config["apply_loader_check"]:
        default_entries.extend([
            f'    "data_check_args": {config["data_check_args"]},',
            f'    "data_check_kwargs": {config["data_check_kwargs"]},'
        ])

    default_entries.extend([
        f'    "model_generate_summary": {config["model_generate_summary"]},',
        f'    "revise_accuracy": {config["revise_accuracy"]},',
        f'    "use_saved_weights": {config["use_saved_weights"]},'
    ])

    if config["revise_accuracy"] or config["use_saved_weights"]:
        default_entries.append(f'    "weights_unmatch_load": {config["weights_unmatch_load"]},')

    default_entries.extend([
        f'    "checkpoint_strict_load": {config["checkpoint_strict_load"]},',
        f'    "log_epochs": {config["log_epochs"]},',
        f'    "train_dynamic_update": {config["train_dynamic_update"]},',
        f'    "final_test": {config["final_test"]},',
        f'    "results_analysis": {config["results_analysis"]}'
    ])

    default_str = "\n".join(default_entries)
    if default_str.endswith(','):
        default_str = default_str.rsplit(',', 1)[0] + default_str.rsplit(',', 1)[1]

    script_content = f"""# Version: {__version__}
import argparse

from workflow import create_workflow

DEFAULT = {{
{default_str}
}}

def parse_args():
    parser = argparse.ArgumentParser(description="Workflow Executor: To execute the generated workflow.")
    parser.add_argument("-l", "--load", action="store_true",
                        help="Enable revise_accuracy and use_saved_weights")
    parser.add_argument("-d", "--device", type=int, default=DEFAULT["device_index"],
                        help="Set device index")
    return parser.parse_args()

def parse_apply():
    args = parse_args()
    if args.load:
        DEFAULT["revise_accuracy"] = True
        DEFAULT["use_saved_weights"] = True
    if args.device != DEFAULT["device_index"]:
        DEFAULT["device_index"] = args.device

if __name__ == "__main__":
    parse_apply()
    workflow_executors = create_workflow(**DEFAULT)

    for executor in workflow_executors:
        executor.run(True, True, 15, 10)
"""

    filename = f"main_{config['config_name'].replace('.yaml', '')}.py"
    with open(filename, "w") as f:
        f.write(script_content)
    print("\033[92mConfiguration script saved as", filename, "You can run it directly.\033[0m")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Workflow Generator: A script that allows you to generate custom-configured scripts based on their needs for creating and running workflows.")
    main_group = parser.add_mutually_exclusive_group()
    main_group.add_argument("-g", "--generate", action="store_true",
                            help="Start the configuration process to generate a script.")
    main_group.add_argument("-i", "--info", action="store_true",
                            help="Show information about the project and exit.")
    main_group.add_argument("-a", "--args", action="store_true",
                            help="Show the meaning of the configurable parameters.")
    main_group.add_argument("-v", "--version", action="version",
                            version=__version__, help="Show the version and exit.")
    parser.add_argument("-t", "--test", action="store_true",
                        help="Test model from saved weights and output csv results.")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="Plot figures for test results.")
    parser.add_argument("-P", "--pred", nargs='*', type=float, default=False,
                        help="Run prediction from saved model weights with optional threshold(s) (0-1). Multiple values can be provided; if none, defaults to None.")

    try:
        args, unknown = parser.parse_known_args()
    except SystemExit:
        sys.exit(1)

    if unknown:
        parser.error(f"Unknown arguments: {', '.join(unknown)}")

    if args.pred != False:
        if any(not 0 <= v <= 1 for v in args.pred):
            parser.error("All thresholds for --pred (-P) must be floats between 0 and 1.")
        all_flags = [args.generate, args.info, args.args, args.test, args.plot]
        if any(all_flags) or "-v" in sys.argv or "--version" in sys.argv:
            parser.error("The --pred (-P) option cannot be used with any other options.")
        return args, unknown

    has_main_action = any([args.generate, args.info, args.args])
    has_version = "-v" in sys.argv or "--version" in sys.argv
    has_test_action = any([args.test, args.plot])

    if (has_main_action or has_version) and has_test_action:
        parser.error("Main actions (-g/-i/-a/-v) cannot be used with test actions (-t/-p)")

    if not has_main_action and not has_test_action:
        return args, unknown

    return args, unknown


if __name__ == "__main__":
    args, unknown = parse_args()

    if args.generate:
        print(
            "\033[96m\nThis script generates a configuration script for your workflow. Follow the prompts to customize your settings.\n\033[0m")
        config = prompt_for_config()
        print("\n\033[96mFinal Configuration:\033[0m")
        for key, value in config.items():
            print(f"\033[94m{key}\033[0m: {value}")
        confirm = input(
            "\033[94mConfirm and generate script? \033[0m(y/n, \033[96mdefault: y\033[0m):").strip().lower()
        if confirm != 'n':
            generate_main_script(config)
    elif args.info:
        print_info()
    elif args.args:
        print_args()
    elif args.test or args.plot or args.pred != False:
        device_index = None
        if args.test or args.pred != False:
            prompt = "pred" if args.pred != False else "test"
            device_index = int(input(f"Enter \033[94mdevice index\033[0m for model {prompt}: ") or 0)
        workflow_executors = create_workflow(test_only=args.test, plot_only=args.plot, predict=args.pred != False,
                                             thresholds=args.pred if args.pred else None,
                                             device_index=device_index if args.test or args.pred != False else None)
        for executor in workflow_executors:
            executor.run(True, True, 5, 5, pre_instr='get_usr_conf', show_end=False)
    else:
        print(
            "\033[93m[Warning]\033[0m Direct execution is not recommended. Use \033[93m`python main.py -g`\033[0m instead.")
        proceed = input("Do you want to proceed anyway? \033[93m(y/n)\033[0m: ").strip().lower()
        if proceed in ['y', 'yes']:
            workflow_executors = create_workflow(
                config_name="default.yaml",
                device_index=0,
                device_info_verbose=True,
                apply_loader_check=True,
                data_check_args=(3, 0),
                data_check_kwargs={"verbose": False, "calculate_proportion": True},
                model_generate_summary=True,
                revise_accuracy=True,
                use_saved_weights=False,
                weights_unmatch_load=True,
                checkpoint_strict_load=True,
                log_epochs=True,
                train_dynamic_update=False,
                final_test=True,
                results_analysis=True
            )
            for executor in workflow_executors:
                executor.run(True, True, 15, 10)
