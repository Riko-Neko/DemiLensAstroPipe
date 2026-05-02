import csv
import os
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import torchmetrics
from torch import cat
from tqdm import tqdm

from config import ModelSettings


def compute_score_hist_JS_OA(probs, labels, nbins=40, range=(0.0, 1.0), eps=1e-12):
    """
    Compute histogram-based PMFs for positive and negative predicted probabilities,
    plus Jensen-Shannon divergence and Overlap Area (OA).

    Args:
        probs: array-like, shape (N,), predicted probabilities in [0,1].
        labels: array-like of {0,1}, shape (N,).
        nbins: int, number of bins (your case: 40).
        range: tuple, histogram range, default (0,1).
        eps: small value to avoid log(0).

    Returns:
        js_divergence: float (>=0), Jensen-Shannon divergence between p and q (in nats).
        overlap_area: float in [0,1], sum(min(p_i, q_i)) — 越小越好.
    """
    probs = np.asarray(probs).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    assert probs.shape[0] == labels.shape[0], "probs and labels must have same length"

    bins = np.linspace(range[0], range[1], nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    pos_mask = labels == 1
    neg_mask = labels == 0

    # handle edge cases
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        # cannot compute meaningful JS/OA without both classes
        js_divergence = float('nan')
        overlap_area = float('nan')
        return js_divergence, overlap_area

    pos_counts, _ = np.histogram(probs[pos_mask], bins=bins)
    neg_counts, _ = np.histogram(probs[neg_mask], bins=bins)

    # 转为概率质量函数（PMF）
    p = pos_counts.astype(np.float64)
    q = neg_counts.astype(np.float64)

    if p.sum() == 0 or q.sum() == 0:
        js_divergence = float('nan')
        overlap_area = float('nan')
        return js_divergence, overlap_area

    p_hist = p / p.sum()
    q_hist = q / q.sum()

    # Overlap Area (OA)
    overlap_area = float(np.sum(np.minimum(p_hist, q_hist)))

    # Jensen-Shannon divergence (discrete)
    m = 0.5 * (p_hist + q_hist)
    # KL(p||m) + KL(q||m) then *0.5
    kl_pm = np.sum(np.where(p_hist > 0, p_hist * np.log((p_hist + eps) / (m + eps)), 0.0))
    kl_qm = np.sum(np.where(q_hist > 0, q_hist * np.log((q_hist + eps) / (m + eps)), 0.0))
    js_divergence = 0.5 * (kl_pm + kl_qm)

    return js_divergence, overlap_area


def test(model, dataloader, criterion, prob_csv_name=None, fpr_tpr_csv_name=None, auc=True, dynamic_accuracy=True,
         process_bar=None, log_func: Optional[Callable] = None, div_measure=True):
    config = ModelSettings.load_config()
    device = ModelSettings.device
    name = ModelSettings.config_name.rsplit('.', 1)[0]
    output_dir = config['path']['test_output_dir']

    prob_csv_name = prob_csv_name if prob_csv_name is not None else f'{name}_prob.csv' if name != 'default.yaml' else f'{model.__class__.__name__}_prob.csv'
    fpr_tpr_csv_name = fpr_tpr_csv_name if fpr_tpr_csv_name is not None else f'{name}_fpr_tpr.csv' if name != 'default.yaml' else f'{model.__class__.__name__}_fpr_tpr.csv'
    output_dir = Path(output_dir) if output_dir is not None else Path('./result/csv')
    fpr_tpr_output_dir, prob_output_dir = output_dir / 'fpr_tpr_output', output_dir / 'prob_output'

    fpr_tpr_output_dir.mkdir(parents=True, exist_ok=True)
    prob_output_dir.mkdir(parents=True, exist_ok=True)

    fpr_tpr_csv_path, prob_csv_path = fpr_tpr_output_dir / fpr_tpr_csv_name, prob_output_dir / prob_csv_name

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    threshold = 0.5
    trues = []
    preds = []

    if fpr_tpr_csv_path.exists():
        new_fpr_tpr_csv_path = fpr_tpr_output_dir / f"{fpr_tpr_csv_path.stem}_old{fpr_tpr_csv_path.suffix}"
        if new_fpr_tpr_csv_path.exists():
            os.remove(new_fpr_tpr_csv_path)
        fpr_tpr_csv_path.rename(new_fpr_tpr_csv_path)

    if prob_csv_path.exists():
        new_prob_csv_path = prob_output_dir / f"{prob_csv_path.stem}_old{prob_csv_path.suffix}"
        if new_prob_csv_path.exists():
            os.remove(new_prob_csv_path)
        prob_csv_path.rename(new_prob_csv_path)

    with open(prob_csv_path, 'w', newline='') as prob_file, open(fpr_tpr_csv_path, 'w', newline='') as roc_file:
        prob_writer = csv.writer(prob_file)
        prob_writer.writerow(['true_labels', 'predicted_probs'])
        roc_writer = csv.writer(roc_file)
        roc_writer.writerow(['fpr', 'tpr', 'thresholds'])

        test_process_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Evaluating: ',
                                colour='magenta') if process_bar is None else process_bar

        with torch.no_grad():
            for i, (images, labels) in test_process_bar:
                images = images.to(device)
                labels = labels.unsqueeze(1).float().to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.sigmoid(outputs)
                predicted = (probs >= threshold).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                trues.append(labels)
                preds.append(probs)

                if dynamic_accuracy:
                    test_process_bar.set_description(
                        f"Test Accuracy: {100 * correct / total:.4f}%, Loss: {loss:.6f} ==> \033[94mWriting AUROC...\033[0m")

                for true, pred in zip(labels.cpu(), probs.cpu()):
                    prob_writer.writerow([true.item(), pred.item()])

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        tqdm.write(f"Test Loss: {avg_loss:.6f}, Accuracy: {100 * accuracy:.4f}%\033[92m  Writing done.\033[0m ",
                   file=None)

        all_trues, all_preds = cat(trues), cat(preds)

        print("\033[94mWriting ROC...\033[0m", end='')
        roc_curve_metric = torchmetrics.ROC(task='binary', num_classes=1)
        fpr, tpr, thresholds = roc_curve_metric(all_preds, all_trues.int())

        youden_index = tpr - fpr
        max_j_idx = torch.argmax(youden_index)
        best_threshold = thresholds[max_j_idx].item()
        best_fpr = fpr[max_j_idx].item()
        best_tpr = tpr[max_j_idx].item()
        max_j = youden_index[max_j_idx].item()

        for fpr_value, tpr_value, threshold in zip(fpr, tpr, thresholds):
            roc_writer.writerow([fpr_value.item(), tpr_value.item(), threshold.item()])

        print("\033[92m Done.\033[0m")
        tqdm.write(
            f"\033[92mThreshold\033[0m(Youden index)\033[92m: {best_threshold:.6f} \033[0m|\033[92m "f"FPR: {best_fpr:.6f}, TPR: {best_tpr:.6f}, J = {max_j:.6f}\033[0m",
            file=None)

        fpr_np = fpr.cpu().numpy()
        tpr_np = tpr.cpu().numpy()

        # === TPR@FPR ===
        for target_fpr in [0.005, 0.01, 0.05]:  # 0.5%, 1%, 5%
            tpr_at_fpr = float(np.interp(target_fpr, fpr_np, tpr_np))
            threshold_at_fpr = float(np.interp(target_fpr, fpr_np, thresholds.cpu().numpy()))
            tqdm.write(
                f"\033[92mTPR@FPR={target_fpr * 100:.1f}%: {tpr_at_fpr:.6f}, Threshold: {threshold_at_fpr:.6f}\033[0m")

        # === FPR@TPR ===
        for target_tpr in [0.90, 0.95, 0.99]:  # 90%, 95%, 99%
            fpr_at_tpr = float(np.interp(target_tpr, tpr_np, fpr_np))
            threshold_at_tpr = float(np.interp(target_tpr, tpr_np, thresholds.cpu().numpy()))
            tqdm.write(
                f"\033[92mFPR@TPR={target_tpr * 100:.0f}%: {fpr_at_tpr:.6f}, Threshold: {threshold_at_tpr:.6f}\033[0m")

        if auc:
            print("\033[94mCalculating AUC...\033[0m", end='')
            auc_metric = torchmetrics.AUROC(task='binary', num_classes=1)
            auc = auc_metric(all_preds, all_trues)
            print("\033[92m Done.\033[0m")
            tqdm.write(f"\033[92mAUC: {auc:.6f}\033[0m", file=None)
        else:
            auc = None

        if div_measure:
            print("\033[94mCalculating distributional divergence measures...\033[0m", end='')
            JS, OA = compute_score_hist_JS_OA(probs=all_preds.cpu().numpy().ravel(),
                                              labels=all_trues.cpu().numpy().ravel().astype(int), nbins=40)
            print("\033[92m Done.\033[0m")
            tqdm.write(
                f"\033[92mJensen-Shannon Divergence (JS): {JS:.6f} \033[0m|\033[92m Overlap Area (OA): {OA:.6f}\033[0m",
                file=None)

        if log_func is not None:
            log_func(best_threshold, best_fpr, best_tpr, max_j, accuracy, loss, criterion.__class__.__name__, auc)

        return best_threshold, best_fpr, best_tpr, auc





