import csv
import time

import matplotlib.pyplot as plt
import torch

from config import ModelSettings, Mapping

# Global variables for plotting
# interpolate_points = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
interpolate_points = [0.01, 0.99]
colors = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#bcbd22',
    '#17becf'
]
MODEL_NAME_MAP = {
    "DemiLensNet-clftfdia4": "DemiLensNet",
    "DemiLensNet-clftfdia_l": "DemiLensNet-L",
    "DemiLensNet-clftfdia_xl": "DemiLensNet-XL",
    "CLFTNetFDIA-clftfdia4": "DemiLensNet",
    "CLFTNetFDIA-clftfdia_l": "DemiLensNet-L",
    "CLFTNetFDIA-clftfdia_xl": "DemiLensNet-XL"
}


def plot_fpr_tpr(fpr_list, tpr_list, threshold_list, x_int, model_names, colors,
                 fname='./result/fig/fpr_tpr_default.pdf'):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define linestyles for each threshold
    linestyles = ['dashed', 'dotted']
    assert len(x_int) == len(linestyles), "x_int and linestyles must be the same length."

    # Add fake plot lines to show dashed styles in legend
    for i in range(len(x_int)):
        ax.plot([], [], color='black', linestyle=linestyles[i],
                label=f'Threshold = {x_int[i]:.2f}')

    for fpr, tpr, threshold, label, color in zip(
            fpr_list, tpr_list, threshold_list, model_names, colors):
        # Convert to tensors if necessary
        fpr = fpr.clone().detach() if isinstance(fpr, torch.Tensor) else torch.tensor(fpr)
        tpr = tpr.clone().detach() if isinstance(tpr, torch.Tensor) else torch.tensor(tpr)
        # fpr_90_tpr = np.interp(0.9, tpr, fpr)
        threshold = threshold.clone().detach() if isinstance(threshold, torch.Tensor) else torch.tensor(threshold)
        x_int = x_int.clone().detach() if isinstance(x_int, torch.Tensor) else torch.tensor(x_int)

        # Interpolate
        y_fpr = linear_interpolate1d(threshold, fpr, x_int)

        # Plot ROC curve
        ax.plot(fpr.numpy(), tpr.numpy(), lw=2, label=f'{label} ROC curve', color=color)

        # Add labels for each sample point on the ROC curve
        # for i in range(len(x_int)):
        #     plt.text(1.08 * y_fpr[i].item(), 0.985 * y_tpr[i].item(), str(round(x_int[i].item(), 2)), color="g",
        #              fontsize=10)

        # Draw vertical lines for each threshold (no label)
        for i in range(len(x_int)):
            ax.axvline(
                x=y_fpr[i].item(),
                ymin=0, ymax=1,
                linestyle=linestyles[i],
                linewidth=0.8,
                color=color
            )

    # Customize the plot
    ax.set_xscale('log')
    ax.set_xlabel('FPR', fontsize=10)
    ax.set_ylabel('TPR', fontsize=10)
    ax.set_ylim(0.8, 1.02)
    ax.tick_params(labelsize=10)
    ax.grid(linestyle='-.', linewidth=0.1)
    ax.legend(loc='lower right', fontsize=8)

    # Save and close
    fig.savefig(fname)
    plt.close(fig)


def plot_probability_distribution(prob_positive_list, prob_negative_list, model_names, colors,
                                  fnames='./result/fig/prob_distri_default'):
    # Set file names for plots
    if not isinstance(fnames, list):
        fnames = [fnames]
        assert len(model_names) != len(fnames), "Filenames and models don't match."

    # Initialize figure for line plot
    fig_line, ax_line = plt.subplots(figsize=(16, 11))

    bins = torch.linspace(0.0, 1.0, 41)

    # Line plot for all models
    for prob_positive, prob_negative, label, color in zip(prob_positive_list, prob_negative_list, model_names, colors):
        prob_positive = torch.tensor(prob_positive)
        prob_negative = torch.tensor(prob_negative)

        # Compute histogram for positives and negatives
        positive_hist, _ = torch.histogram(prob_positive, bins=bins)
        negative_hist, _ = torch.histogram(prob_negative, bins=bins)

        # Normalize histogram to plot as probability
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        positive_hist = positive_hist.float() / positive_hist.sum()
        negative_hist = negative_hist.float() / negative_hist.sum()

        # Plot line for positives and negatives
        ax_line.plot(bin_centers.numpy(), positive_hist.numpy(), label=f"{label} positives", color=color,
                     linestyle='-', marker='o')
        ax_line.plot(bin_centers.numpy(), negative_hist.numpy(), label=f"{label} negatives", color=color,
                     linestyle='--', marker='x')

        """
        # --- 不确定区（0.3-0.7）平均密度水平线（用于 line 图） ---
        mask = (bin_centers >= 0.3) & (bin_centers <= 0.7)
        uncertain_mean = (positive_hist + negative_hist)[mask].mean().item()
        ax_line.axhline(uncertain_mean, color='gray', linestyle='--', linewidth=1.5,
                        label=f'Uncertain mean (0.3-0.7)={uncertain_mean:.3e}')

        # --- 通过 ROC 计算阈值：TPR@FPR=0.1 与 FPR@TPR=0.9 ---
        pos_np = prob_positive.numpy()
        neg_np = prob_negative.numpy()
        probs_all = np.concatenate([pos_np, neg_np])
        labels_all = np.concatenate([np.ones(len(pos_np)), np.zeros(len(neg_np))])

        fpr, tpr, thr = roc_curve(labels_all, probs_all)

        # 1) 找到在 fpr <= 0.1 下使 tpr 最大的阈值（对应 TPR@FPR=0.1）
        idx_fpr = np.where(fpr <= 0.1)[0]
        if idx_fpr.size > 0:
            idx_choice = idx_fpr[np.argmax(tpr[idx_fpr])]
        else:
            idx_choice = np.argmin(np.abs(fpr - 0.1))
        thr_fpr_0_1 = float(thr[idx_choice])

        # 2) 找到在 tpr >= 0.9 下使 fpr 最小的阈值（对应 FPR@TPR=0.9）
        idx_tpr = np.where(tpr >= 0.9)[0]
        if idx_tpr.size > 0:
            idx_choice2 = idx_tpr[np.argmin(fpr[idx_tpr])]
        else:
            idx_choice2 = np.argmin(np.abs(tpr - 0.9))
        thr_tpr_0_9 = float(thr[idx_choice2])

        # 绘制竖线（line plot 上）
        ax_line.axvline(thr_fpr_0_1, color=color, linestyle='-', linewidth=2,
                        label=f'Thr (FPR<=0.1)={thr_fpr_0_1:.3f}')
        ax_line.axvline(thr_tpr_0_9, color=color, linestyle=':', linewidth=2,
                        label=f'Thr (TPR>=0.9)={thr_tpr_0_9:.3f}')
        """

    # Customize the line plot
    ax_line.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_line.set_xlabel(f'$P$', size=16)
    ax_line.set_ylabel('Probability', size=16)
    ax_line.set_xlim(0, 1)
    ax_line.set_yscale('log')
    ax_line.legend(loc='upper center', fontsize=10, ncol=2)
    ax_line.grid(linestyle='--', alpha=0.5)

    # Save the line plot
    fig_line.savefig(f'{fnames[0]}_line.pdf')
    plt.close(fig_line)

    # Generate individual bar plots for each model
    for prob_positive, prob_negative, label, color, fname in zip(prob_positive_list, prob_negative_list, model_names,
                                                                 colors, fnames):
        fig_bar, ax_bar = plt.subplots(figsize=(8, 5.5))

        prob_positive = torch.tensor(prob_positive)
        prob_negative = torch.tensor(prob_negative)

        # Plot the histogram for positives and negatives separately
        ax_bar.hist(prob_positive.numpy(), bins=bins.numpy(), histtype='bar', rwidth=0.9, log=True, alpha=0.7,
                    edgecolor='black', label=f"{label} positives", color=color)
        ax_bar.hist(prob_negative.numpy(), bins=bins.numpy(), histtype='bar', rwidth=0.9, log=True, alpha=0.7,
                    edgecolor='black', label=f"{label} negatives", color='gray')

        # Customize the bar plot
        ax_bar.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax_bar.set_xlabel(f'$P$', size=16)
        ax_bar.set_ylabel('Number', size=16)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_ylim(0.8, 100000)
        ax_bar.legend(loc='upper center', fontsize=12)
        ax_bar.grid(linestyle='--', alpha=0.5)

        # Save the bar plot for this model
        fig_bar.savefig(f'{fname}_{label}_bar.pdf')
        plt.close(fig_bar)


def plot_completeness(prob_positive_list, num_positive_list, model_names, colors,
                      fname='./result/fig/completeness_default.pdf'):
    # Initialize figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot completeness for each model with different color
    for prob_positive, num_positive, label, color in zip(prob_positive_list, num_positive_list, model_names, colors):
        prob_positive = torch.tensor(prob_positive)
        num_positive = torch.tensor(num_positive)
        x = torch.arange(100) * 0.01
        y = torch.zeros_like(x)

        for i in range(len(x)):
            y[i] = (torch.sum(prob_positive >= x[i]) / num_positive)

        # Plot the completeness with the specified color
        plt.plot(x.numpy(), y.numpy(), lw=1, label=f'{label} completeness', color=color)

    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
    ax.set_ylim(0.95, 1.0)
    plt.xlabel(f'$P$', size=10)
    plt.ylabel('Completeness', size=10)

    # Adding a legend to match each model with the color
    plt.legend(loc=(0.07, 0.05), fontsize=8)
    plt.grid(linestyle='-', linewidth=0.1)

    # Save the plot
    plt.savefig(fname)
    plt.close()


def linear_interpolate1d(x_tensor, y_tensor, x_int):
    """
        Performs 1D linear interpolation to estimate y values at specified x coordinates.

        Args:
        - x_tensor (torch.Tensor): A 1D tensor containing the known x coordinates of the data points.
        - y_tensor (torch.Tensor): A 1D tensor containing the known y values corresponding to the x coordinates.
        - x_int (torch.Tensor): A 1D tensor containing the x coordinates at which to perform the interpolation.

        Returns:
        - torch.Tensor: A 1D tensor containing the interpolated y values corresponding to the provided x coordinates in x_int.

        Example:
        >>> x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        >>> y = torch.tensor([0.0, 2.0, 4.0, 6.0])
        >>> x_int = torch.tensor([0.5, 1.5, 2.5])
        >>> y_interpolated = linear_interpolate1d(x, y, x_int)
        >>> print(y_interpolated)
        tensor([0.5000, 2.5000, 4.5000])
        """

    def get_points_closest(x_tensor, y_tensor, x):
        if x_tensor.shape[0] != y_tensor.shape[0]:
            raise ValueError("Wrong csv index, file might be corrupted.")

        x_left = x_tensor[x_tensor < x]
        if x_left.any():
            xi = x_left.max()
            yi = y_tensor[-len(x_left)]
        else:
            xi = x_tensor.min()
            yi = y_tensor[x_tensor.argmin()]

        x_right = x_tensor[x_tensor > x]
        if x_right.any():
            xj = x_right.min()
            yj = y_tensor[len(x_right) - 1]
        else:
            xj = x_tensor.max()
            yj = y_tensor[x_tensor.argmax()]

        return xi, yi, xj, yj

    def interpolate_unit(x1_tensor, y1_tensor, x2_tensor, y2_tensor, x):
        y = torch.lerp(y1_tensor, y2_tensor, (x - x1_tensor) / (x2_tensor - x1_tensor))
        return y.item()

    y_int = torch.zeros_like(x_int)
    for i in range(len(x_int)):
        xi, yi, xj, yj = get_points_closest(x_tensor, y_tensor, x_int[i])
        y_int[i] = interpolate_unit(xi, yi, xj, yj, x_int[i])
    return y_int


def plot_data_reader(model_names, fpr_tpr_csv_paths, prob_csv_paths):
    def read_fpr_tpr(csv_file):
        fpr, tpr, threshold = [], [], []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fpr.append(float(row['fpr']))
                tpr.append(float(row['tpr']))
                threshold.append(float(row['thresholds']))
        return fpr, tpr, threshold

    def read_prob(csv_file):
        prob_pos = []
        prob_neg = []

        with open(csv_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                true_label = float(row['true_labels'])
                predicted_prob = float(row['predicted_probs'])

                if true_label == 1.0:
                    prob_pos.append(predicted_prob)
                elif true_label == 0.0:
                    prob_neg.append(predicted_prob)

        return prob_pos, prob_neg

    time.sleep(0.5)
    print("\033[94mReading csv...\033[0m")
    # Read data for both models
    fpr_list, tpr_list, threshold_list = [], [], []
    prob_positive_list, prob_negative_list = [], []
    num_positive_list, num_negative_list = [], []

    for model_name, fpr_tpr_csv, prob_csv in zip(model_names, fpr_tpr_csv_paths, prob_csv_paths):
        print(f"Reading {model_name} data...")

        fpr, tpr, threshold = read_fpr_tpr(fpr_tpr_csv)
        prob_positive, prob_negative = read_prob(prob_csv)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        threshold_list.append(threshold)
        prob_positive_list.append(prob_positive)
        prob_negative_list.append(prob_negative)
        num_positive_list.append(len(prob_positive))
        num_negative_list.append(len(prob_negative))

    return fpr_list, tpr_list, threshold_list, prob_positive_list, prob_negative_list, num_positive_list, num_negative_list


def plot_main(model_names, fpr_tpr_csv_paths, prob_csv_paths, x_int, colors, fpr_tpr_fname, prob_distri_fnames,
              completeness_fname):
    # Read and arrange data from csv files
    fpr_list, tpr_list, threshold_list, prob_positive_list, prob_negative_list, num_positive_list, num_negative_list = plot_data_reader(
        model_names, fpr_tpr_csv_paths, prob_csv_paths)

    # Plot the ROC curve and thresholds for each model
    print("\033[94mPlotting fpr_tpr...\033[0m", end='')
    plot_fpr_tpr(fpr_list, tpr_list, threshold_list, x_int, model_names, colors, fname=fpr_tpr_fname)
    print("\033[92mDone!\033[0m")

    # Plot the probability distribution for each model
    print("\033[94mPlotting probability distribution...\033[0m", end='')
    plot_probability_distribution(prob_positive_list, prob_negative_list, model_names, colors,
                                  fnames=prob_distri_fnames)
    print("\033[92mDone!\033[0m")

    # Plot the completeness for each model
    print("\033[94mPlotting completeness...\033[0m", end='')
    plot_completeness(prob_positive_list, num_positive_list, model_names, colors, fname=completeness_fname)
    print("\033[92mDone!\033[0m")

    print("\033[92mPlot done.\033[0m")


def plot_from_workflow():
    config_name = ModelSettings.config_name
    model_name = ModelSettings.load_config()['model']['model_name']
    csv_dir = ModelSettings.load_config()['path']['test_output_dir']
    output_dir = ModelSettings.load_config()['path']['plot_output_dir']
    name_cls_str = Mapping.MODEL_MAPPING.get(model_name, model_name)
    if isinstance(name_cls_str, type):
        model_name = [name_cls_str.__name__]
    else:
        model_name = [name_cls_str]

    fpr_tpr_csv_paths = [f'{csv_dir}/fpr_tpr_output/{config_name}_fpr_tpr.csv']
    prob_csv_paths = [f'{csv_dir}/prob_output/{config_name}_prob.csv']
    fpr_tpr_fname = f'{output_dir}/fpr_tpr_{config_name}.pdf'
    prob_distri_fnames = [f'{output_dir}/prob_distri_{config_name}']
    completeness_fname = f'{output_dir}/completeness_{config_name}.pdf'

    x_int = interpolate_points

    plot_main(model_name, fpr_tpr_csv_paths, prob_csv_paths, x_int, colors, fpr_tpr_fname, prob_distri_fnames,
              completeness_fname)


def plot_from_config():
    def _get_unique_names(headers, names):
        from collections import defaultdict

        count = defaultdict(int)
        for item in names:
            count[item] += 1

        for i in range(len(names)):
            if count[names[i]] > 1:
                names[i] = f"{names[i]}-{headers[i]}"
        return names

    def _map_model_names(model_names, name_mapping=None):

        if not name_mapping:
            return model_names

        return [
            name_mapping.get(name, name)
            for name in model_names
        ]

    def str_connector(names):
        return '_'.join(names)

    output_dirs = []
    csv_dirs = []
    config_names = []
    model_names = []
    paths = ModelSettings.user_config_lib if ModelSettings.user_config_lib is not None else ModelSettings.get_config_paths()

    print("\033[94mLoading chosen configs...\033[0m")
    for path in paths:
        ModelSettings.init_config(path, full_path=True)
        config = ModelSettings.load_config()
        output_dir = ModelSettings.load_config()['path']['plot_output_dir']
        csv_dir = ModelSettings.load_config()['path']['test_output_dir']
        model_name = config['model']['model_name']

        output_dirs.append(output_dir)
        csv_dirs.append(csv_dir)
        config_names.append(ModelSettings.config_name)
        name_cls_str = Mapping.MODEL_MAPPING.get(model_name, model_name)
        if isinstance(name_cls_str, type):
            model_names.append(name_cls_str.__name__)
        else:
            model_names.append(name_cls_str)

    config_names_sub = [name.rsplit('.', 1)[0] for name in config_names]
    model_names = _get_unique_names(config_names_sub, model_names)
    model_names = _map_model_names(model_names, name_mapping=MODEL_NAME_MAP)
    long_name = str_connector(config_names_sub)
    fpr_tpr_csv_paths = [f'{csv_dirs[i]}/fpr_tpr_output/{name}_fpr_tpr.csv' for i, name in enumerate(config_names_sub)]
    prob_csv_paths = [f'{csv_dirs[i]}/prob_output/{name}_prob.csv' for i, name in enumerate(config_names_sub)]
    fpr_tpr_fname = f'{output_dirs[0]}/fpr_tpr_{long_name}.pdf'
    prob_distri_fnames = [f'{output_dirs[i]}/prob_distri_{long_name}' for i in range(len(output_dirs))]
    completeness_fname = f'{output_dirs[0]}/completeness_{long_name}.pdf'

    x_int = interpolate_points

    plot_main(model_names, fpr_tpr_csv_paths, prob_csv_paths, x_int, colors, fpr_tpr_fname, prob_distri_fnames,
              completeness_fname)


if __name__ == '__main__':
    print(
        "[\033[93mWarning\033[0m] The standalone use of \033[93mplot.py\033[0m is not recommended. It is highly advised to use the \033[93mmain script\033[0m with \033[93m-p\033[0m or \033[93m--plot\033[0m option, which is safer.")
    time.sleep(3)
    plot_from_config()
