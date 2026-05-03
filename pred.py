import csv
import math
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from config import ModelSettings


def predict(model, dataloader_predict, threshold=0.5, process_bar=None, label_instr: str = None):
    config = ModelSettings.load_config()
    device = ModelSettings.device
    name = ModelSettings.config_name.rsplit('.', 1)[0]
    output_dir = config['path']['pred_output_dir']
    output_type = config['path']['pred_output_type']

    _, path = next(iter(dataloader_predict))
    data_dir = Path(path[0]).parent
    data_dir_name = data_dir.name
    path_default = data_dir.parent
    output_dir = path_default if output_dir is None else Path(output_dir)
    output_dir = output_dir / f'{name}_{data_dir_name}_pred'
    if output_type == 'csv':
        output_dir = output_dir.with_suffix('.csv')
    elif output_type != 'file':
        raise ValueError(f"[\033[91mError\033[0m] Unsupported output type: {output_type}. Available types: 'file', 'csv'.")
    print(
        f"\033[94mPredicting samples in {data_dir} to {output_dir}\033[0m ==> \033[96m{len(dataloader_predict.dataset)} prediction samples\033[0m")

    predict_process_bar = tqdm(enumerate(dataloader_predict), total=len(dataloader_predict),
                               desc='Predicting: ',
                               colour='green') if process_bar is None else process_bar

    if output_type == 'file':
        if output_dir.exists():
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                print(f"[\033[91mError\033[0m] Error clearing directory [\033[91m{output_dir}\033[0m]: {e}")

        output_dir.mkdir(parents=True, exist_ok=True)
        pos_dir = output_dir / 'pos'
        neg_dir = output_dir / 'neg'
        pos_dir.mkdir(exist_ok=True)
        neg_dir.mkdir(exist_ok=True)
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            try:
                output_dir.unlink()
            except Exception as e:
                print(f"[\033[91mError\033[0m] Error clearing file [\033[91m{output_dir}\033[0m]: {e}")

    model.eval()

    # Stats for optional label evaluation
    TP = FP = FN = 0
    all_probs = []
    if output_type == 'csv':
        with open(output_dir, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['sample', 'prob'])

            with torch.no_grad():
                for i, (images, paths) in predict_process_bar:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs).float()

                    for j in range(len(paths)):
                        prob = probs[j].item()
                        all_probs.append(prob)
                        pred_file = Path(paths[j])

                        if not pred_file.exists():
                            print(f"[\033[91mError\033[0m] Source file [\033[91m{pred_file}\033[0m] does not exist.")
                            continue

                        csv_writer.writerow([pred_file.name, prob])

                        is_positive = prob >= threshold
                        if label_instr == 'all_1':
                            if is_positive:
                                TP += 1
                            else:
                                FN += 1
                        elif label_instr == 'all_0':
                            if is_positive:
                                FP += 1
    else:
        with torch.no_grad():
            for i, (images, paths) in predict_process_bar:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).float()

                for j in range(len(paths)):
                    prob = probs[j].item()
                    all_probs.append(prob)
                    pred_file = Path(paths[j])

                    if not pred_file.exists():
                        print(f"[\033[91mError\033[0m] Source file [\033[91m{pred_file}\033[0m] does not exist.")
                        continue

                    new_filename = f"{pred_file.stem}_prob_{prob:.2f}{pred_file.suffix}"

                    is_positive = prob >= threshold
                    try:
                        if is_positive:
                            shutil.copy(pred_file, pos_dir / new_filename)
                        else:
                            shutil.copy(pred_file, neg_dir / new_filename)
                    except Exception as e:
                        print(
                            f"[\033[91mError\033[0m] Error copying file [\033[91m{pred_file}\033[0m] to destination: {e}")

                    # Optional metrics evaluation if label_mode is active
                    if label_instr == 'all_1':
                        if is_positive:
                            TP += 1
                        else:
                            FN += 1
                    elif label_instr == 'all_0':
                        if is_positive:
                            FP += 1

    if label_instr == 'all_1':
        n = len(all_probs)
        if n == 0:
            print("[\033[93mWarning\033[0m] No probabilities collected; cannot compute threshold.")
        else:
            try:
                user_input = input(
                    "[\033[92mInfo\033[0m] Enter target completeness (0–1 decimal), or press Enter to skip: ").strip()
            except Exception:
                user_input = ''

            if user_input == '':
                print("Skipped completeness-to-threshold computation.")
            else:
                try:
                    target = float(user_input)
                    if not (0.0 <= target <= 1.0):
                        raise ValueError("out of range")
                except Exception:
                    print(
                        f"[\033[91mError\033[0m] Invalid input ({user_input}), must be a float between 0 and 1. Skipped.")
                    target = None

                if target is not None:
                    # Sort probabilities descending
                    probs_sorted = sorted(all_probs, reverse=True)

                    # Find smallest threshold such that fraction(prob >= t) >= target
                    if target == 0.0:
                        threshold_for_target = 1.0
                    else:
                        k = math.ceil(target * n)
                        if k <= 0:
                            threshold_for_target = 1.0
                        elif k > n:
                            threshold_for_target = probs_sorted[-1]
                        else:
                            threshold_for_target = probs_sorted[k - 1]

                    # Display results
                    print(f"[\033[92mInfo\033[0m]\nTarget completeness: {target:.4f}, total samples: {n}.")
                    print(f"Threshold achieving ≥ {target:.4f} recall: {threshold_for_target:.6f}")
                    actual_recall = sum(p >= threshold_for_target for p in all_probs) / n
                    print(f"Actual achieved completeness: {actual_recall:.6f}")

    # Calculate precision and recall if needed
    if label_instr in ('all_1', 'all_0'):
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        return output_dir, precision, recall, FNR

    return output_dir, None, None, None
