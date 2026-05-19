import os
import sys
import time
import math
import random
import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from model import model

DATA_DIR = os.environ.get("PULSEDB_DIR", os.path.join(THIS_DIR, "..", "PulseDB"))
SAVE_ROOT = os.environ.get("SAVE_ROOT", os.path.join(THIS_DIR, "..", "saved_model"))
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(SAVE_ROOT, "run_" + RUN_TIMESTAMP)

TARGET_LEN = 1024
BATCH_SIZE = 256
EPOCHS = 100
PATIENCE = 25
LR = 1e-3
WEIGHT_DECAY = 1e-5
NUM_RUNS = 10
SEEDS = list(range(NUM_RUNS))
NUM_WORKERS = 4
PIN_MEMORY = torch.cuda.is_available()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_logger(path: str) -> logging.Logger:
    ensure_dir(os.path.dirname(path))
    logger = logging.getLogger(path)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def minmax_2d(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    return (x - x_min) / (x_max - x_min + eps)


def as_segments(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    if x.shape[1] < TARGET_LEN:
        raise ValueError(f"Segment length {x.shape[1]} is shorter than {TARGET_LEN}")
    if x.shape[1] > TARGET_LEN:
        x = x[:, :TARGET_LEN]
    return x.astype(np.float32, copy=False)


def load_data(subset_dir: str, logger: logging.Logger, name: str) -> Tuple[np.ndarray, np.ndarray]:
    ppg_dir = os.path.join(subset_dir, "ppg")
    abp_dir = os.path.join(subset_dir, "abp")
    patient_ids = sorted(file_name[:-4] for file_name in os.listdir(ppg_dir) if file_name.endswith(".npy"))

    ppg_list: List[np.ndarray] = []
    abp_list: List[np.ndarray] = []
    skipped = 0

    iterator = patient_ids
    if tqdm is not None:
        iterator = tqdm(patient_ids, desc=f"Loading {name}", dynamic_ncols=True)

    for patient_id in iterator:
        ppg_path = os.path.join(ppg_dir, patient_id + ".npy")
        abp_path = os.path.join(abp_dir, patient_id + ".npy")
        if not os.path.exists(abp_path):
            skipped += 1
            continue

        try:
            ppg = as_segments(np.load(ppg_path))
            abp = as_segments(np.load(abp_path))
        except Exception:
            skipped += 1
            continue

        n = min(len(ppg), len(abp))
        if n == 0:
            skipped += 1
            continue

        ppg_list.append(minmax_2d(ppg[:n]))
        abp_list.append(abp[:n])

    if not ppg_list:
        raise RuntimeError(f"No data was loaded from {subset_dir}.")

    ppg = np.concatenate(ppg_list, axis=0).astype(np.float32, copy=False)
    abp = np.concatenate(abp_list, axis=0).astype(np.float32, copy=False)

    logger.info(f"{name}_segments: {len(ppg)}")
    logger.info(f"{name}_skipped_patients: {skipped}")
    logger.info(f"{name}_ppg_shape: {ppg.shape}")
    logger.info(f"{name}_abp_shape: {abp.shape}")
    return ppg, abp


class PulseDataset(Dataset):
    def __init__(self, ppg: np.ndarray, abp: np.ndarray):
        self.ppg = ppg
        self.abp = abp

    def __len__(self) -> int:
        return len(self.ppg)

    def __getitem__(self, idx: int):
        ppg = torch.from_numpy(self.ppg[idx]).float().unsqueeze(0)
        abp = torch.from_numpy(self.abp[idx]).float().unsqueeze(0)
        return ppg, abp


def make_loader(dataset: Dataset, shuffle: bool, drop_last: bool, seed: int = None) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
    }
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        kwargs["generator"] = generator
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def create_loaders(
    train_data: Tuple[np.ndarray, np.ndarray],
    valid_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = make_loader(PulseDataset(*train_data), True, False, seed)
    valid_loader = make_loader(PulseDataset(*valid_data), False, False)
    test_loader = make_loader(PulseDataset(*test_data), False, False)
    return train_loader, valid_loader, test_loader


def run_epoch(
    net: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer = None,
) -> float:
    is_train = optimizer is not None
    net.train() if is_train else net.eval()

    loss_sum = 0.0
    n_samples = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for ppg, abp in loader:
            ppg = ppg.to(device, non_blocking=True)
            abp = abp.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            pred = net(ppg)
            loss = criterion(pred, abp)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = abp.size(0)
            loss_sum += float(loss.item()) * batch_size
            n_samples += batch_size

    n = max(n_samples, 1)
    return loss_sum / n


def evaluate(net: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, Any]:
    net.eval()

    loss_sum = 0.0
    n_samples = 0
    n_points = 0
    err_sum = 0.0
    err_sq_sum = 0.0
    abs_err_sum = 0.0

    with torch.no_grad():
        for ppg, abp in loader:
            ppg = ppg.to(device, non_blocking=True)
            abp = abp.to(device, non_blocking=True)
            pred = net(ppg)
            loss = criterion(pred, abp)

            batch_size = abp.size(0)
            loss_sum += float(loss.item()) * batch_size
            n_samples += batch_size

            y_true = abp[:, 0, :]
            y_pred = pred[:, 0, :]
            err = y_pred - y_true
            err_sum += float(err.sum().item())
            err_sq_sum += float((err ** 2).sum().item())
            abs_err_sum += float(torch.abs(err).sum().item())
            n_points += int(err.numel())

    mean_err = err_sum / max(n_points, 1)
    mse = err_sq_sum / max(n_points, 1)

    metrics: Dict[str, Any] = {
        "loss_mse": loss_sum / max(n_samples, 1),
        "abp_mae": abs_err_sum / max(n_points, 1),
        "abp_rmse": math.sqrt(mse),
        "abp_me": mean_err,
        "abp_sd": math.sqrt(max(mse - mean_err ** 2, 0.0)),
        "n_samples": int(n_samples),
    }
    return metrics


def save_state_dict(net: nn.Module, path: str) -> None:
    base_model = net.module if isinstance(net, nn.DataParallel) else net
    torch.save(base_model.state_dict(), path)


def load_state_dict(net: nn.Module, path: str, device: torch.device) -> None:
    base_model = net.module if isinstance(net, nn.DataParallel) else net
    state_dict = torch.load(path, map_location=device)
    base_model.load_state_dict(state_dict)


def flatten_columns(columns) -> List[str]:
    output = []
    for col in columns:
        if isinstance(col, tuple):
            output.append("_".join(str(item) for item in col if item != ""))
        else:
            output.append(str(col))
    return output


def main() -> None:
    ensure_dir(SAVE_DIR)
    logger = setup_logger(os.path.join(SAVE_DIR, "train.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"data_dir: {DATA_DIR}")
    logger.info(f"save_dir: {SAVE_DIR}")
    logger.info(f"device: {device}")
    logger.info(f"batch_size: {BATCH_SIZE}")
    logger.info(f"epochs: {EPOCHS}")
    logger.info(f"patience: {PATIENCE}")
    logger.info(f"optimizer: Adam")
    logger.info(f"learning_rate: {LR}")
    logger.info(f"weight_decay: {WEIGHT_DECAY}")
    logger.info(f"loss: MSELoss")

    train_data = load_data(os.path.join(DATA_DIR, "train"), logger, "train")
    valid_data = load_data(os.path.join(DATA_DIR, "valid"), logger, "valid")
    test_data = load_data(os.path.join(DATA_DIR, "test"), logger, "test")
    results: List[Dict[str, Any]] = []
    results_path = os.path.join(SAVE_DIR, "test_results.csv")
    aggregate_path = os.path.join(SAVE_DIR, "test_results_aggregate.csv")

    for seed in SEEDS:
        set_seed(seed)
        train_loader, valid_loader, test_loader = create_loaders(train_data, valid_data, test_data, seed)
        run_logger = setup_logger(os.path.join(SAVE_DIR, f"seed{seed}.log"))

        net = model()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        net = net.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_valid_loss = float("inf")
        best_epoch = 0
        patience_count = 0
        best_path = os.path.join(SAVE_DIR, f"seed{seed}_best.pt")
        history_path = os.path.join(SAVE_DIR, f"seed{seed}_history.csv")
        history: List[Dict[str, Any]] = []
        start_time = time.time()

        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()
            train_loss = run_epoch(
                net, train_loader, criterion, device, optimizer
            )
            valid_loss = run_epoch(
                net, valid_loader, criterion, device
            )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "epoch_time_sec": time.time() - epoch_start,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(history_path, index=False)

            run_logger.info(
                f"epoch {epoch:03d}/{EPOCHS} "
                f"train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}"
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                patience_count = 0
                save_state_dict(net, best_path)
            else:
                patience_count += 1

            if patience_count >= PATIENCE:
                break

        train_time_min = (time.time() - start_time) / 60.0
        load_state_dict(net, best_path, device)
        test_metrics = evaluate(net, test_loader, criterion, device)
        result = {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
            "train_time_min": train_time_min,
            **test_metrics,
        }
        results.append(result)
        pd.DataFrame(results).to_csv(results_path, index=False)

        results_df = pd.DataFrame(results)
        metric_cols = [
            col
            for col in results_df.select_dtypes(include=[np.number]).columns
            if col not in ["seed", "best_epoch", "train_time_min"]
        ]
        if metric_cols:
            aggregate_df = results_df[metric_cols].agg(["mean", "std"]).reset_index()
            aggregate_df.to_csv(aggregate_path, index=False)

        for key, value in result.items():
            run_logger.info(f"{key}: {value}")

        del net, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
