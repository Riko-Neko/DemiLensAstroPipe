import argparse
import copy
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from astropy.io import fits
from torch.amp import GradScaler, autocast
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from config import ModelSettings
from utils import BuilderManager, TrainingPipelineBuilder


PROFILE_PRESETS = {
    "default": {
        "pos_fraction": 0.10,
        "neg_ratio": 3,
        "hard_neg_fraction": 0.70,
        "batch_size_cap": 16,
        "probe_size_min": 256,
        "probe_size_mul": 16,
        "stage1_head_lr_scale": 1.0,
        "stage1_head_lr_min": 5e-5,
        "stage1_head_lr_max": 1e-4,
        "stage2_head_lr_ratio": 0.30,
        "stage2_backbone_lr_ratio": 0.10,
        "weight_decay_cap": 1e-4,
        "warmup_rounds": 12,
        "lr_decay_patience": 40,
        "reset_patience": 80,
        "pos_weight_scale": 1.25,
        "pos_weight_min": 1.5,
        "pos_weight_max": 4.0,
    },
    "aggressive": {
        "pos_fraction": 0.10,
        "neg_ratio": 2,
        "hard_neg_fraction": 0.85,
        "batch_size_cap": 8,
        "probe_size_min": 512,
        "probe_size_mul": 32,
        "stage1_head_lr_scale": 2.0,
        "stage1_head_lr_min": 1e-4,
        "stage1_head_lr_max": 2e-4,
        "stage2_head_lr_ratio": 0.50,
        "stage2_backbone_lr_ratio": 0.20,
        "weight_decay_cap": 5e-5,
        "warmup_rounds": 6,
        "lr_decay_patience": 24,
        "reset_patience": 48,
        "pos_weight_scale": 1.0,
        "pos_weight_min": 1.25,
        "pos_weight_max": 3.0,
    },
}


class FineTunePathDataset(Dataset):
    def __init__(self, paths, labels, img_size, adaptation_mode="padding", train=False,
                 norm=False, mean=None, std=None):
        self.paths = list(paths)
        self.labels = list(labels)
        self.img_size = img_size
        self.adaptation_mode = adaptation_mode
        self.train = train
        self.norm = norm
        self.mean = [0.485, 0.456, 0.406] if mean is None else mean
        self.std = [0.229, 0.224, 0.225] if std is None else std
        self.transform = self._build_transform()

    def _build_transform(self):
        transform_list = [v2.ToImage()]

        if self.adaptation_mode == "resizing":
            if self.train:
                transform_list.extend([
                    v2.RandomResize(
                        min_size=self.img_size,
                        max_size=max(self.img_size, int(self.img_size * 1.05)),
                        interpolation=Image.BICUBIC,
                    ),
                    v2.RandomCrop((self.img_size, self.img_size), pad_if_needed=True, padding_mode="reflect"),
                ])
            else:
                transform_list.extend([
                    v2.Resize((self.img_size, self.img_size), interpolation=Image.BICUBIC),
                ])
        elif self.adaptation_mode == "padding":
            if self.train:
                transform_list.extend([
                    v2.CenterCrop(int(self.img_size * 1.1)),
                    v2.RandomCrop((self.img_size, self.img_size), pad_if_needed=True, padding_mode="reflect"),
                ])
            else:
                transform_list.append(v2.CenterCrop(int(self.img_size)))
        elif self.adaptation_mode == "original":
            pass
        else:
            raise ValueError('Invalid adaptation mode. (available modes: "resizing", "padding", "original")')

        if self.train:
            transform_list.extend([
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomRotation(180),
            ])

        transform_list.append(v2.ToDtype(torch.float32, scale=True))

        if self.norm:
            transform_list.append(v2.Normalize(mean=self.mean, std=self.std))

        return v2.Compose(transform_list)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        img = self._load_image(path)
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

    @staticmethod
    def pixel_filter(image):
        image[image < 0] = 0.0
        return image

    def _load_image(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".fits":
            return self.load_fits(path)
        if ext == ".png":
            return self.load_png(path)
        raise ValueError(f"Unsupported file format: {ext}")

    def load_png(self, path):
        image = Image.open(path).convert("RGB")
        return image

    def load_fits(self, path):
        bands = []
        reference_shape = None

        with fits.open(path) as hdul:
            for i in range(len(hdul)):
                data = hdul[i].data
                if data is None:
                    continue

                if isinstance(data, np.ndarray) and data.dtype.names is not None:
                    for name in data.dtype.names:
                        col = data[name][0]
                        if isinstance(col, (np.ndarray, list)):
                            size = int(len(col))
                            data = np.array(col, dtype=np.float32).reshape(size, size)
                            break

                if reference_shape is None:
                    reference_shape = data.shape
                if data.shape != reference_shape:
                    continue

                band_data = self.pixel_filter(data)
                if len(bands) == 1:
                    band_data = np.sqrt(band_data)

                if band_data.dtype.byteorder != "=":
                    band_data = band_data.byteswap().view(band_data.dtype.newbyteorder("="))

                bands.append(torch.from_numpy(band_data).float())

        if not bands:
            raise ValueError(f"Failed to load FITS bands from {path}")

        combined_tensor = torch.cat(bands)
        max_value = 1e-13 if (max_value := torch.max(combined_tensor)).item() == 0 else max_value
        for idx in range(len(bands)):
            bands[idx] /= max_value

        return torch.stack(bands, dim=0)


def print_header(title):
    print(f"\n\033[94m{title}\033[0m")


def print_kv(label, value, color="96"):
    print(f"{label}: \033[{color}m{value}\033[0m")


def prompt_path(prompt_text, default=None, must_exist=True):
    while True:
        suffix = f" (\033[96mdefault: {default}\033[0m)" if default else ""
        value = input(f"{prompt_text}{suffix}: ").strip()
        if value == "" and default is not None:
            value = default
        if value == "":
            print("[\033[91mError\033[0m] Path cannot be empty.")
            continue
        if must_exist and not os.path.exists(value):
            print(f"[\033[91mError\033[0m] Path \033[91m{value}\033[0m does not exist.")
            continue
        return value


def prompt_float(prompt_text, default, min_value=None, max_value=None):
    while True:
        raw = input(f"{prompt_text} (\033[96mdefault: {default}\033[0m): ").strip()
        if raw == "":
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                print("[\033[91mError\033[0m] Invalid float.")
                continue
        if min_value is not None and value < min_value:
            print(f"[\033[91mError\033[0m] Value should be >= {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"[\033[91mError\033[0m] Value should be <= {max_value}.")
            continue
        return value


def prompt_int(prompt_text, default, min_value=None):
    while True:
        raw = input(f"{prompt_text} (\033[96mdefault: {default}\033[0m): ").strip()
        if raw == "":
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("[\033[91mError\033[0m] Invalid integer.")
                continue
        if min_value is not None and value < min_value:
            print(f"[\033[91mError\033[0m] Value should be >= {min_value}.")
            continue
        return value


def find_files_in_dirs(dirs):
    files = []
    for directory in dirs or []:
        if not os.path.exists(directory):
            continue
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in (".png", ".fits"):
                files.append(path)
    return sorted(files)


def classify_negative_paths(paths):
    hard_paths = []
    base_paths = []
    for path in paths:
        lower = path.lower()
        if any(token in lower for token in ("kids", "lrg", "real", "hard", "dr4")):
            hard_paths.append(path)
        else:
            base_paths.append(path)
    if not hard_paths:
        return list(paths), []
    return hard_paths, base_paths


def threshold_at_target_recall(probs, target_recall):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.size == 0:
        return 0.0, 0.0
    keep = max(1, int(math.ceil(target_recall * probs.size)))
    sorted_probs = np.sort(probs)[::-1]
    threshold = float(sorted_probs[keep - 1])
    achieved = float(np.mean(probs >= threshold))
    return threshold, achieved


def mean_bottom_fraction(probs, fraction):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.size == 0:
        return 0.0
    count = max(1, int(math.ceil(probs.size * fraction)))
    return float(np.mean(np.sort(probs)[:count]))


def detect_head_prefixes(model):
    preferred = [
        "fc",
        "classifier",
        "head",
        "heads",
        "mlp_head",
        "last_linear",
        "logits",
    ]
    prefixes = [name for name in preferred if hasattr(model, name)]
    if prefixes:
        return prefixes

    named_children = list(model.named_children())
    if not named_children:
        return []
    return [named_children[-1][0]]


def split_named_parameters(model, head_prefixes):
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in head_prefixes):
            head_params.append((name, param))
        else:
            backbone_params.append((name, param))
    if not head_params:
        named_params = list(model.named_parameters())
        split_idx = max(1, int(len(named_params) * 0.85))
        backbone_params = named_params[:split_idx]
        head_params = named_params[split_idx:]
    return head_params, backbone_params


def set_trainable_state(head_params, backbone_params, head_only):
    for _, param in head_params:
        param.requires_grad = True
    for _, param in backbone_params:
        param.requires_grad = not head_only


def build_optimizer(head_params, backbone_params, head_only, head_lr, backbone_lr, weight_decay):
    if head_only:
        params = [param for _, param in head_params if param.requires_grad]
        return torch.optim.AdamW(params, lr=head_lr, weight_decay=weight_decay)

    return torch.optim.AdamW([
        {"params": [param for _, param in backbone_params if param.requires_grad], "lr": backbone_lr},
        {"params": [param for _, param in head_params if param.requires_grad], "lr": head_lr},
    ], weight_decay=weight_decay)


def update_optimizer_lrs(optimizer, head_only, head_lr, backbone_lr):
    if head_only:
        optimizer.param_groups[0]["lr"] = head_lr
        return
    if len(optimizer.param_groups) == 1:
        optimizer.param_groups[0]["lr"] = head_lr
        return
    optimizer.param_groups[0]["lr"] = backbone_lr
    optimizer.param_groups[1]["lr"] = head_lr


def choose_negative_indices(rng, hard_indices, base_indices, neg_take, hard_fraction):
    hard_pick = 0
    if hard_indices:
        hard_pick = min(len(hard_indices), int(round(neg_take * hard_fraction)))
    base_pick = neg_take - hard_pick

    selected = []
    if hard_pick > 0:
        selected.extend(rng.sample(hard_indices, hard_pick) if hard_pick < len(hard_indices) else list(hard_indices))

    base_pool = base_indices
    if base_pick > 0 and base_pool:
        selected.extend(rng.sample(base_pool, base_pick) if base_pick < len(base_pool) else list(base_pool))

    if len(selected) < neg_take:
        remain_pool = hard_indices + base_indices
        if remain_pool:
            missing = neg_take - len(selected)
            selected.extend(rng.choices(remain_pool, k=missing))

    rng.shuffle(selected)
    return selected[:neg_take]


def run_inference(model, dataloader, device, amp_enabled, amp_device_type):
    model.eval()
    probs = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            with autocast(device_type=amp_device_type, enabled=amp_enabled):
                logits = model(images)
            batch_probs = torch.sigmoid(logits).detach().float().view(-1).cpu().numpy().tolist()
            probs.extend(batch_probs)
    return np.asarray(probs, dtype=np.float64)


def evaluate_metrics(model, p19_loader, neg_loader, device, amp_enabled, amp_device_type, target_threshold, target_recall):
    p19_probs = run_inference(model, p19_loader, device, amp_enabled, amp_device_type)
    neg_probs = run_inference(model, neg_loader, device, amp_enabled, amp_device_type) if neg_loader is not None else np.asarray([])

    p19_recall_at_target = float(np.mean(p19_probs >= target_threshold)) if p19_probs.size else 0.0
    threshold_target_recall, threshold_target_recall_actual = threshold_at_target_recall(p19_probs, target_recall)
    neg_pass_rate = float(np.mean(neg_probs >= target_threshold)) if neg_probs.size else 0.0
    p19_mean = float(np.mean(p19_probs)) if p19_probs.size else 0.0
    p19_tail_mean = mean_bottom_fraction(p19_probs, max(0.1, 1.0 - target_recall))

    metrics = {
        "p19_recall_at_target": p19_recall_at_target,
        "p19_target_recall_threshold": threshold_target_recall,
        "p19_target_recall_actual": threshold_target_recall_actual,
        "neg_pass_rate_at_target": neg_pass_rate,
        "p19_mean_prob": p19_mean,
        "p19_tail_mean_prob": p19_tail_mean,
        "p19_probs": p19_probs,
        "neg_probs": neg_probs,
    }
    return metrics


def better_metrics(new_metrics, best_metrics, tol=1e-8):
    if best_metrics is None:
        return True
    new_key = (
        new_metrics["p19_recall_at_target"],
        new_metrics["p19_target_recall_threshold"],
        -new_metrics["neg_pass_rate_at_target"],
        new_metrics["p19_tail_mean_prob"],
        new_metrics["p19_mean_prob"],
    )
    old_key = (
        best_metrics["p19_recall_at_target"],
        best_metrics["p19_target_recall_threshold"],
        -best_metrics["neg_pass_rate_at_target"],
        best_metrics["p19_tail_mean_prob"],
        best_metrics["p19_mean_prob"],
    )
    for new_value, old_value in zip(new_key, old_key):
        if new_value > old_value + tol:
            return True
        if new_value < old_value - tol:
            return False
    return False


def load_weights(model, load_path, device):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Weights file not found: {load_path}")
    print(f"\n\033[94mLoading weights from {load_path}\033[0m")
    state_dict = torch.load(load_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)


def maybe_resume_weights(base_weight_path, finetune_weight_path):
    if not os.path.exists(finetune_weight_path):
        return base_weight_path

    while True:
        answer = input(
            f"Detected existing fine-tuned weights:\n\033[96m{finetune_weight_path}\033[0m\nUse it as start point? (\033[96my/n\033[0m, default: n): "
        ).strip().lower()
        if answer in ("", "n"):
            return base_weight_path
        if answer == "y":
            return finetune_weight_path
        print("[\033[91mError\033[0m] Please input y or n.")


def prepare_fixed_probe(paths, count, seed):
    rng = random.Random(seed)
    indices = list(range(len(paths)))
    if len(indices) <= count:
        return indices
    return rng.sample(indices, count)


def count_overlap(paths_a, paths_b):
    names_a = {Path(path).name for path in paths_a}
    names_b = {Path(path).name for path in paths_b}
    return len(names_a & names_b)


def extract_coords_from_name(path):
    stem = Path(path).stem
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", stem)
    if len(matches) < 2:
        return None
    try:
        return round(float(matches[-2]), 6), round(float(matches[-1]), 6)
    except ValueError:
        return None


def find_coordinate_overlap(paths_a, paths_b):
    coords_a = {}
    coords_b = {}

    for path in paths_a:
        coord = extract_coords_from_name(path)
        if coord is not None:
            coords_a.setdefault(coord, []).append(path)

    for path in paths_b:
        coord = extract_coords_from_name(path)
        if coord is not None:
            coords_b.setdefault(coord, []).append(path)

    overlap_coords = sorted(set(coords_a) & set(coords_b))
    examples = []
    for coord in overlap_coords[:5]:
        examples.append((coord, coords_a[coord][0], coords_b[coord][0]))
    return overlap_coords, examples


def enforce_no_data_leakage(target_paths, p19_paths, neg_paths):
    target_names = {Path(path).name for path in target_paths}
    p19_names = {Path(path).name for path in p19_paths}
    neg_names = {Path(path).name for path in neg_paths}

    overlap_tp = sorted(target_names & p19_names)
    overlap_np = sorted(neg_names & p19_names)
    coord_overlap_tp, coord_examples_tp = find_coordinate_overlap(target_paths, p19_paths)
    coord_overlap_np, coord_examples_np = find_coordinate_overlap(neg_paths, p19_paths)

    if overlap_tp or coord_overlap_tp:
        preview = overlap_tp[:5]
        if coord_examples_tp:
            preview.extend([f"{Path(a).name} <-> {Path(b).name}" for _, a, b in coord_examples_tp[:3]])
        raise ValueError(
            "[Error] Data leakage detected between fine-tune positives and P19. "
            f"Examples: {preview}"
        )

    if overlap_np or coord_overlap_np:
        preview = overlap_np[:5]
        if coord_examples_np:
            preview.extend([f"{Path(a).name} <-> {Path(b).name}" for _, a, b in coord_examples_np[:3]])
        raise ValueError(
            "[Error] Data leakage detected between negative pool and P19. "
            f"Examples: {preview}"
        )


def train_one_round(model, dataloader, device, optimizer, scaler, pos_weight, amp_enabled, amp_device_type):
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    model.train()
    total_loss = 0.0
    total_items = 0

    process_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Fine-tuning", colour="red")
    for batch_idx, (images, labels) in process_bar:
        optimizer.zero_grad(set_to_none=True)

        images = images.to(device, non_blocking=True)
        labels = labels.view(-1, 1).to(device, non_blocking=True)

        with autocast(device_type=amp_device_type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        avg_loss = total_loss / max(1, total_items)

        process_bar.set_description(
            f"\033[38;5;22m\033[47mFine-tune\033[0m, \033[92mBatch {batch_idx + 1}/{len(dataloader)}\033[0m, \033[96mLoss: {avg_loss:.6f}\033[0m"
        )

    return total_loss / max(1, total_items)


def build_dataloader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone P19-oriented fine-tuning script. It does not modify workflow or project modules."
    )
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config yaml.")
    parser.add_argument("-d", "--device", type=int, default=0, help="Device index.")
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        choices=sorted(PROFILE_PRESETS.keys()),
        help="Fine-tune profile preset.",
    )
    parser.add_argument("--target-data", type=str, default=None, help="Positive fine-tune sample directory.")
    parser.add_argument("--target-threshold", type=float, default=None, help="Primary evaluation threshold.")
    parser.add_argument("--target-recall", type=float, default=None, help="Secondary recall target.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    print_header("Standalone Fine-tune | P19 Recall Driven")

    if args.config is None:
        config_path = prompt_path("Path of the config file")
    else:
        config_path = args.config

    ModelSettings.init_config(config_path, full_path=True)
    ModelSettings.set_device(device_index=args.device, verbose=True, info_verbose=False)
    device = ModelSettings.device
    config = ModelSettings.load_config()
    config_stem = str(Path(ModelSettings.config_name).stem)
    profile = PROFILE_PRESETS[args.profile]
    builder_manager = BuilderManager()
    training_pipeline = TrainingPipelineBuilder()
    test_log = training_pipeline.read_test_log()

    if args.target_data is None:
        target_data_dir = prompt_path("Path of the fine-tune positive samples")
    else:
        target_data_dir = args.target_data

    target_threshold = args.target_threshold
    if target_threshold is None:
        logged_threshold = test_log.get("best_threshold", None)
        default_threshold = 0.5 if logged_threshold is None else float(logged_threshold)
        target_threshold = prompt_float("Target threshold for P19 recall@target", default_threshold, 0.0, 1.0)

    target_recall = args.target_recall
    if target_recall is None:
        target_recall = prompt_float("Target recall for threshold tracking", 0.90, 0.0, 1.0)

    print_header("Preparing Data")
    target_paths = find_files_in_dirs([target_data_dir])
    if not target_paths:
        raise FileNotFoundError(f"No files found in {target_data_dir}")

    p19_dirs = config["path"].get("data_dir_pred", [])
    if not p19_dirs:
        raise ValueError("Current config does not define data_dir_pred. P19 evaluation set is unavailable.")
    p19_paths = find_files_in_dirs(p19_dirs)
    if not p19_paths:
        raise FileNotFoundError(f"No files found in P19 test dirs: {p19_dirs}")

    neg_dirs = config["path"].get("neg_dir_train", [])
    neg_paths = find_files_in_dirs(neg_dirs)
    if not neg_paths:
        raise FileNotFoundError(f"No negative files found in config neg_dir_train: {neg_dirs}")

    enforce_no_data_leakage(target_paths, p19_paths, neg_paths)

    hard_neg_paths, base_neg_paths = classify_negative_paths(neg_paths)
    print_kv("Profile", args.profile)
    print_kv("Fine-tune positives", len(target_paths))
    print_kv("P19 positives", len(p19_paths))
    print_kv("Negative pool", len(neg_paths))
    print_kv("Hard negatives", len(hard_neg_paths))
    print_kv("Base negatives", len(base_neg_paths))

    img_size = config["data"]["image_size"]
    adaptation_mode = config["data"]["adaptation_mode"]
    norm = config["data"]["norm"]
    mean = config["data"]["mean"]
    std = config["data"]["std"]
    num_workers = min(int(config["data"]["num_workers"]), 4)

    pos_fraction = profile["pos_fraction"]
    pos_take = max(1, int(math.ceil(len(target_paths) * pos_fraction)))
    neg_ratio = profile["neg_ratio"]
    neg_take = max(1, min(len(neg_paths), pos_take * neg_ratio))
    hard_neg_fraction = profile["hard_neg_fraction"] if base_neg_paths else 1.0
    batch_size = min(profile["batch_size_cap"], max(4, pos_take))
    probe_size = min(max(profile["probe_size_min"], pos_take * profile["probe_size_mul"]), len(neg_paths))

    print_kv("Per-round positive subset", pos_take)
    print_kv("Per-round negative subset", neg_take)
    print_kv("Batch size", batch_size)
    print_kv("Target threshold", f"{target_threshold:.6f}")
    print_kv("Target recall", f"{target_recall:.4f}")

    p19_dataset = FineTunePathDataset(
        p19_paths,
        [1] * len(p19_paths),
        img_size=img_size,
        adaptation_mode=adaptation_mode,
        train=False,
        norm=norm,
        mean=mean,
        std=std,
    )
    p19_loader = build_dataloader(
        p19_dataset,
        batch_size=min(32, max(4, batch_size)),
        num_workers=num_workers,
        shuffle=False,
    )

    probe_indices = prepare_fixed_probe(hard_neg_paths if hard_neg_paths else neg_paths, probe_size, args.seed)
    probe_source = hard_neg_paths if hard_neg_paths else neg_paths
    probe_paths = [probe_source[i] for i in probe_indices]
    neg_probe_dataset = FineTunePathDataset(
        probe_paths,
        [0] * len(probe_paths),
        img_size=img_size,
        adaptation_mode=adaptation_mode,
        train=False,
        norm=norm,
        mean=mean,
        std=std,
    )
    neg_probe_loader = build_dataloader(
        neg_probe_dataset,
        batch_size=min(64, max(8, batch_size * 2)),
        num_workers=num_workers,
        shuffle=False,
    )

    print_header("Building Model")
    model = builder_manager.model_builder(generate_summary=False)

    weights_dir = Path(config["path"]["weights_dir"])
    base_weight_path = str(weights_dir / f"{config_stem}_weights.pth")
    finetune_weight_path = str(weights_dir / f"{config_stem}_p19_finetune_best.pth")
    load_path = maybe_resume_weights(base_weight_path, finetune_weight_path)
    load_weights(model, load_path, device)
    model.to(device)

    head_prefixes = detect_head_prefixes(model)
    head_params, backbone_params = split_named_parameters(model, head_prefixes)
    print_kv("Detected head modules", ", ".join(head_prefixes) if head_prefixes else "fallback")
    print_kv("Head params", sum(param.numel() for _, param in head_params))
    print_kv("Backbone params", sum(param.numel() for _, param in backbone_params))

    base_lr = float(config["train"]["learning_rate"])
    base_wd = float(config["train"]["weight_decay"])
    stage1_head_lr = min(
        max(base_lr * profile["stage1_head_lr_scale"], profile["stage1_head_lr_min"]),
        profile["stage1_head_lr_max"],
    )
    stage2_head_lr = max(stage1_head_lr * profile["stage2_head_lr_ratio"], 1e-5)
    stage2_backbone_lr = max(stage2_head_lr * profile["stage2_backbone_lr_ratio"], 1e-6)
    weight_decay = min(base_wd, profile["weight_decay_cap"]) if base_wd > 0 else 1e-6
    warmup_rounds = profile["warmup_rounds"]
    lr_decay_patience = profile["lr_decay_patience"]
    reset_patience = profile["reset_patience"]
    amp_enabled = torch.cuda.is_available()
    amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"

    print_header("Fine-tune Strategy")
    print_kv("Stage-1", f"head-only | {warmup_rounds} rounds | lr={stage1_head_lr:.2e}")
    print_kv("Stage-2", f"full-model | head_lr={stage2_head_lr:.2e} | backbone_lr={stage2_backbone_lr:.2e}")
    print_kv("Weight decay", f"{weight_decay:.2e}")
    print_kv("AMP", amp_enabled)

    set_trainable_state(head_params, backbone_params, head_only=True)
    optimizer = build_optimizer(
        head_params=head_params,
        backbone_params=backbone_params,
        head_only=True,
        head_lr=stage1_head_lr,
        backbone_lr=stage2_backbone_lr,
        weight_decay=weight_decay,
    )
    scaler = GradScaler(device=amp_device_type, enabled=amp_enabled)

    initial_metrics = evaluate_metrics(
        model=model,
        p19_loader=p19_loader,
        neg_loader=neg_probe_loader,
        device=device,
        amp_enabled=amp_enabled,
        amp_device_type=amp_device_type,
        target_threshold=target_threshold,
        target_recall=target_recall,
    )
    best_metrics = copy.deepcopy(initial_metrics)
    best_state_dict = copy.deepcopy(model.state_dict())
    best_round = 0
    no_improve_rounds = 0

    print_header("Initial Metrics")
    print_kv("P19 Recall@Target", f"{initial_metrics['p19_recall_at_target']:.4f}")
    print_kv("Threshold@TargetRecall", f"{initial_metrics['p19_target_recall_threshold']:.6f}")
    print_kv("Actual Recall@TargetRecallThr", f"{initial_metrics['p19_target_recall_actual']:.4f}")
    print_kv("Neg Pass Rate@Target", f"{initial_metrics['neg_pass_rate_at_target']:.4f}")
    print_kv("P19 Mean Prob", f"{initial_metrics['p19_mean_prob']:.6f}")
    print_kv("P19 Tail Mean", f"{initial_metrics['p19_tail_mean_prob']:.6f}")
    torch.save(best_state_dict, finetune_weight_path)
    print(f"\033[92mSaved initial fine-tune slot to\033[0m \033[96m{finetune_weight_path}\033[0m")

    target_rng = random.Random(args.seed + 1)
    neg_index_all = list(range(len(neg_paths)))
    hard_neg_indices = [idx for idx, path in enumerate(neg_paths) if path in set(hard_neg_paths)]
    base_neg_indices = [idx for idx in neg_index_all if idx not in set(hard_neg_indices)]

    print_header("Start Fine-tuning")
    round_idx = 0
    stage = 1

    try:
        while True:
            round_idx += 1
            if stage == 1 and round_idx == warmup_rounds + 1:
                stage = 2
                set_trainable_state(head_params, backbone_params, head_only=False)
                optimizer = build_optimizer(
                    head_params=head_params,
                    backbone_params=backbone_params,
                    head_only=False,
                    head_lr=stage2_head_lr,
                    backbone_lr=stage2_backbone_lr,
                    weight_decay=weight_decay,
                )
                print(f"\n\033[94mStage switch\033[0m ==> \033[96mFull-model fine-tuning\033[0m")

            pos_indices = target_rng.sample(list(range(len(target_paths))), pos_take) if pos_take < len(target_paths) else list(range(len(target_paths)))
            neg_indices = choose_negative_indices(
                rng=target_rng,
                hard_indices=hard_neg_indices,
                base_indices=base_neg_indices,
                neg_take=neg_take,
                hard_fraction=hard_neg_fraction,
            )

            round_pos_paths = [target_paths[idx] for idx in pos_indices]
            round_neg_paths = [neg_paths[idx] for idx in neg_indices]
            round_dataset = FineTunePathDataset(
                round_pos_paths + round_neg_paths,
                [1] * len(round_pos_paths) + [0] * len(round_neg_paths),
                img_size=img_size,
                adaptation_mode=adaptation_mode,
                train=True,
                norm=norm,
                mean=mean,
                std=std,
            )
            train_loader = build_dataloader(
                round_dataset,
                batch_size=min(batch_size, len(round_dataset)),
                num_workers=num_workers,
                shuffle=True,
            )

            pos_weight = min(
                profile["pos_weight_max"],
                max(
                    profile["pos_weight_min"],
                    (len(round_neg_paths) / max(1, len(round_pos_paths))) * profile["pos_weight_scale"],
                ),
            )
            current_head_lr = optimizer.param_groups[-1]["lr"]
            current_backbone_lr = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 1 else 0.0

            print(f"\n\033[94mRound \033[96m{round_idx}\033[0m | "
                  f"\033[94mStage:\033[0m \033[96m{'head-only' if stage == 1 else 'full-model'}\033[0m | "
                  f"\033[94mPos:\033[0m \033[96m{len(round_pos_paths)}\033[0m | "
                  f"\033[94mNeg:\033[0m \033[96m{len(round_neg_paths)}\033[0m")
            print(f"\033[94mLearning Rate\033[0m ==> head: \033[96m{current_head_lr:.2e}\033[0m"
                  + (f", backbone: \033[96m{current_backbone_lr:.2e}\033[0m" if stage == 2 else ""))
            print(f"\033[94mLoss Weight\033[0m ==> pos_weight: \033[96m{pos_weight:.3f}\033[0m")

            round_loss = train_one_round(
                model=model,
                dataloader=train_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                pos_weight=pos_weight,
                amp_enabled=amp_enabled,
                amp_device_type=amp_device_type,
            )

            metrics = evaluate_metrics(
                model=model,
                p19_loader=p19_loader,
                neg_loader=neg_probe_loader,
                device=device,
                amp_enabled=amp_enabled,
                amp_device_type=amp_device_type,
                target_threshold=target_threshold,
                target_recall=target_recall,
            )

            improved = better_metrics(metrics, best_metrics)
            print(f"\033[96mRound Loss: {round_loss:.6f}\033[0m")
            print(f"\033[92mP19 Recall@Target:\033[0m \033[96m{metrics['p19_recall_at_target']:.4f}\033[0m "
                  f"(best: \033[96m{best_metrics['p19_recall_at_target']:.4f}\033[0m)")
            print(f"\033[92mThreshold@TargetRecall:\033[0m \033[96m{metrics['p19_target_recall_threshold']:.6f}\033[0m "
                  f"(best: \033[96m{best_metrics['p19_target_recall_threshold']:.6f}\033[0m)")
            print(f"\033[92mNeg Pass Rate@Target:\033[0m \033[96m{metrics['neg_pass_rate_at_target']:.4f}\033[0m "
                  f"(best: \033[96m{best_metrics['neg_pass_rate_at_target']:.4f}\033[0m)")
            print(f"\033[92mP19 Tail Mean:\033[0m \033[96m{metrics['p19_tail_mean_prob']:.6f}\033[0m "
                  f"(best: \033[96m{best_metrics['p19_tail_mean_prob']:.6f}\033[0m)")

            if improved:
                best_metrics = copy.deepcopy(metrics)
                best_state_dict = copy.deepcopy(model.state_dict())
                best_round = round_idx
                no_improve_rounds = 0
                torch.save(best_state_dict, finetune_weight_path)
                print(f"\033[41m \033[42m \033[43m \033[44m \033[45m \033[46m \033[47m"
                      f"\033[38;5;22mP19 target metric improved at round {round_idx}, saving weights to {finetune_weight_path}"
                      f"\033[46m \033[45m \033[44m \033[43m \033[42m \033[41m \033[0m")
            else:
                no_improve_rounds += 1
                print(f"\033[93mNo improvement\033[0m ==> \033[96m{no_improve_rounds}\033[0m consecutive round(s)")

            if no_improve_rounds > 0 and no_improve_rounds % lr_decay_patience == 0:
                if stage == 1:
                    stage1_head_lr = max(stage1_head_lr * 0.5, 1e-5)
                    update_optimizer_lrs(optimizer, head_only=True, head_lr=stage1_head_lr, backbone_lr=0.0)
                    print(f"\033[93mLR Decay\033[0m ==> new head lr: \033[96m{stage1_head_lr:.2e}\033[0m")
                else:
                    stage2_head_lr = max(stage2_head_lr * 0.5, 5e-6)
                    stage2_backbone_lr = max(stage2_backbone_lr * 0.5, 5e-7)
                    update_optimizer_lrs(
                        optimizer,
                        head_only=False,
                        head_lr=stage2_head_lr,
                        backbone_lr=stage2_backbone_lr,
                    )
                    print(f"\033[93mLR Decay\033[0m ==> head: \033[96m{stage2_head_lr:.2e}\033[0m, "
                          f"backbone: \033[96m{stage2_backbone_lr:.2e}\033[0m")

            if no_improve_rounds > 0 and no_improve_rounds % reset_patience == 0:
                model.load_state_dict(best_state_dict, strict=False)
                print(f"\033[93mModel Reset\033[0m ==> restored best state from round \033[96m{best_round}\033[0m")

    except KeyboardInterrupt:
        print(f"\n\033[94mInterrupted by user.\033[0m")
        model.load_state_dict(best_state_dict, strict=False)
        torch.save(best_state_dict, finetune_weight_path)
        print(f"\033[92mBest fine-tuned weights kept at\033[0m \033[96m{finetune_weight_path}\033[0m")
        print(f"\033[92mBest round:\033[0m \033[96m{best_round}\033[0m")
        print(f"\033[92mBest P19 Recall@Target:\033[0m \033[96m{best_metrics['p19_recall_at_target']:.4f}\033[0m")
        print(f"\033[92mBest Threshold@TargetRecall:\033[0m \033[96m{best_metrics['p19_target_recall_threshold']:.6f}\033[0m")


if __name__ == "__main__":
    main()
