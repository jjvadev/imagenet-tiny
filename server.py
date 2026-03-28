import argparse
import csv
import json
import math
import os
import pickle
import socket
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# =========================================================
# Terminal styling
# =========================================================
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


def supports_color():
    return sys.stdout.isatty()


USE_COLOR = supports_color()


def color(text, tone):
    if not USE_COLOR:
        return text
    return f"{tone}{text}{C.RESET}"


def ts():
    return datetime.now().strftime("%H:%M:%S")


def log(msg, level="INFO"):
    tones = {
        "INFO": C.CYAN,
        "OK": C.GREEN,
        "WARN": C.YELLOW,
        "ERR": C.RED,
        "STEP": C.MAGENTA,
        "METRIC": C.BLUE,
    }
    level_str = color(f"{level:>6}", tones.get(level, C.WHITE))
    print(f"[{ts()}] {level_str} | {msg}", flush=True)


def banner():
    line = "=" * 88
    print(color(line, C.MAGENTA))
    print(color(" FEDERATED TINY-IMAGENET SERVER ".center(88), C.BOLD + C.CYAN))
    print(color(line, C.MAGENTA), flush=True)


# =========================================================
# Socket helpers
# =========================================================
def send_msg(sock, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("!Q", len(payload))
    sock.sendall(header + payload)


def recv_exact(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed by peer")
        data.extend(chunk)
    return bytes(data)


def recv_msg(sock):
    header = recv_exact(sock, 8)
    msg_len = struct.unpack("!Q", header)[0]
    payload = recv_exact(sock, msg_len)
    return pickle.loads(payload)


# =========================================================
# Model and metrics
# =========================================================
def build_model(num_classes=200):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def state_dict_to_cpu(state_dict):
    out = {}
    for k, v in state_dict.items():
        out[k] = v.detach().cpu() if torch.is_tensor(v) else v
    return out


def weighted_average_state_dict(worker_payloads):
    total_samples = sum(p["num_samples"] for p in worker_payloads)
    if total_samples <= 0:
        raise ValueError("No valid samples received for aggregation")

    ref_state = worker_payloads[0]["state_dict"]
    agg = {}

    for key in ref_state.keys():
        ref_tensor = ref_state[key]
        if torch.is_tensor(ref_tensor) and torch.is_floating_point(ref_tensor):
            acc = torch.zeros_like(ref_tensor, dtype=torch.float32)
            for payload in worker_payloads:
                weight = payload["num_samples"] / total_samples
                acc += payload["state_dict"][key].float() * weight
            agg[key] = acc.to(ref_tensor.dtype)
        else:
            agg[key] = ref_tensor.clone() if torch.is_tensor(ref_tensor) else ref_tensor

    return agg


def topk_accuracy(logits, targets, topk=(1, 5)):
    with torch.no_grad():
        maxk = min(max(topk), logits.size(1))
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            k = min(k, logits.size(1))
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.item())
        return res


def select_device(prefer_mps=True):
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =========================================================
# Dataset
# =========================================================
class TinyImageNetDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        wnids_path = self.root / "wnids.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(f"No existe: {wnids_path}")

        with open(wnids_path, "r", encoding="utf-8") as f:
            self.wnids = [line.strip() for line in f if line.strip()]

        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.samples = []

        if split == "train":
            train_dir = self.root / "train"
            for wnid in self.wnids:
                img_dir = train_dir / wnid / "images"
                if not img_dir.exists():
                    continue
                for img_path in sorted(img_dir.glob("*.JPEG")):
                    self.samples.append((str(img_path), self.class_to_idx[wnid]))

        elif split == "val":
            val_dir = self.root / "val"
            ann_file = val_dir / "val_annotations.txt"
            if not ann_file.exists():
                raise FileNotFoundError(f"No existe: {ann_file}")

            mapping = {}
            with open(ann_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        img_name, wnid = parts[0], parts[1]
                        mapping[img_name] = wnid

            img_dir = val_dir / "images"
            for img_name, wnid in mapping.items():
                img_path = img_dir / img_name
                if img_path.exists() and wnid in self.class_to_idx:
                    self.samples.append((str(img_path), self.class_to_idx[wnid]))
        else:
            raise ValueError("split debe ser 'train' o 'val'")

        if not self.samples:
            raise RuntimeError(f"No se encontraron imágenes en {root} [{split}]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target


def get_val_loader(data_dir, batch_size, num_workers):
    normalize = transforms.Normalize(
        mean=[0.4802, 0.4481, 0.3975],
        std=[0.2302, 0.2265, 0.2262],
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = TinyImageNetDataset(data_dir, split="val", transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        bsz = targets.size(0)
        top1, top5 = topk_accuracy(logits, targets, topk=(1, 5))

        total_loss += loss.item() * bsz
        total_top1 += top1
        total_top5 += top5
        total_samples += bsz

        if batch_idx % 20 == 0 or batch_idx == len(loader):
            cur_loss = total_loss / total_samples
            cur_top1 = total_top1 / total_samples
            cur_top5 = total_top5 / total_samples
            log(
                f"Validation progress {batch_idx}/{len(loader)} | "
                f"loss={cur_loss:.4f} | acc={cur_top1:.4f} | top5={cur_top5:.4f}",
                "STEP",
            )

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
        "samples": total_samples,
    }


# =========================================================
# Reporting
# =========================================================
def make_run_dir(base_dir="runs"):
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"federated_run_{ts_now}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_metrics_csv(history, out_path):
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_metrics_json(history, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def plot_curve(x, ys, labels, title, ylabel, out_path):
    plt.figure(figsize=(10, 5.5))
    for y, label in zip(ys, labels):
        plt.plot(x, y, marker="o", label=label)
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_all_plots(history, run_dir):
    if not history:
        return

    rounds = [h["round"] for h in history]

    plot_curve(
        rounds,
        [[h["train_loss"] for h in history], [h["val_loss"] for h in history]],
        ["Train Loss", "Validation Loss"],
        "Loss per Round",
        "Loss",
        run_dir / "loss_curve.png",
    )

    plot_curve(
        rounds,
        [[h["train_acc"] for h in history], [h["val_acc"] for h in history]],
        ["Train Accuracy", "Validation Accuracy"],
        "Top-1 Accuracy per Round",
        "Accuracy",
        run_dir / "accuracy_top1_curve.png",
    )

    plot_curve(
        rounds,
        [[h["train_top5"] for h in history], [h["val_top5"] for h in history]],
        ["Train Top-5", "Validation Top-5"],
        "Top-5 Accuracy per Round",
        "Accuracy",
        run_dir / "accuracy_top5_curve.png",
    )

    plot_curve(
        rounds,
        [[h["throughput"] for h in history]],
        ["Throughput"],
        "Throughput per Round",
        "samples/s",
        run_dir / "throughput_curve.png",
    )

    plot_curve(
        rounds,
        [[h["round_time"] for h in history]],
        ["Round Time"],
        "Round Time per Round",
        "seconds",
        run_dir / "round_time_curve.png",
    )


def create_notebook(run_dir):
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Federated Tiny-ImageNet Report\n",
                    "\n",
                    "Notebook generado automáticamente.\n",
                    "\n",
                    "Incluye:\n",
                    "- carga de métricas\n",
                    "- tabla completa\n",
                    "- mejores rondas\n",
                    "- gráficas\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from pathlib import Path\n",
                    "\n",
                    f'run_dir = Path(r"{str(run_dir)}")\n',
                    'with open(run_dir / "metrics.json", "r", encoding="utf-8") as f:\n',
                    "    history = json.load(f)\n",
                    "df = pd.DataFrame(history)\n",
                    "df\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('Última ronda:')\n",
                    "display(df.tail(1))\n",
                    "print('Mejor val_acc:')\n",
                    "display(df.loc[[df['val_acc'].idxmax()]])\n",
                    "print('Mejor val_top5:')\n",
                    "display(df.loc[[df['val_top5'].idxmax()]])\n",
                    "print('Menor val_loss:')\n",
                    "display(df.loc[[df['val_loss'].idxmin()]])\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(10,5))\n",
                    "plt.plot(df['round'], df['train_loss'], marker='o', label='Train Loss')\n",
                    "plt.plot(df['round'], df['val_loss'], marker='o', label='Val Loss')\n",
                    "plt.title('Loss per Round')\n",
                    "plt.xlabel('Round')\n",
                    "plt.ylabel('Loss')\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "plt.legend()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(10,5))\n",
                    "plt.plot(df['round'], df['train_acc'], marker='o', label='Train Acc')\n",
                    "plt.plot(df['round'], df['val_acc'], marker='o', label='Val Acc')\n",
                    "plt.title('Top-1 Accuracy per Round')\n",
                    "plt.xlabel('Round')\n",
                    "plt.ylabel('Accuracy')\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "plt.legend()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(10,5))\n",
                    "plt.plot(df['round'], df['train_top5'], marker='o', label='Train Top5')\n",
                    "plt.plot(df['round'], df['val_top5'], marker='o', label='Val Top5')\n",
                    "plt.title('Top-5 Accuracy per Round')\n",
                    "plt.xlabel('Round')\n",
                    "plt.ylabel('Accuracy')\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "plt.legend()\n",
                    "plt.show()\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(run_dir / "federated_report.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)


def write_summary(history, run_dir, args, total_wall_time):
    if not history:
        return

    first = history[0]
    last = history[-1]
    best_acc = max(history, key=lambda x: x["val_acc"])
    best_top5 = max(history, key=lambda x: x["val_top5"])
    best_loss = min(history, key=lambda x: x["val_loss"])

    lines = []
    lines.append("FEDERATED TRAINING SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Dataset root: {args.data_dir}")
    lines.append(f"Run dir: {run_dir}")
    lines.append(f"Device: {args.device_name}")
    lines.append(f"Rounds requested: {args.rounds}")
    lines.append(f"Rounds completed: {len(history)}")
    lines.append(f"Workers expected: {args.num_workers}")
    lines.append(f"Local epochs: {args.local_epochs}")
    lines.append(f"Learning rate: {args.lr}")
    lines.append(f"Train batch size: {args.train_batch_size}")
    lines.append(f"Val batch size: {args.val_batch_size}")
    lines.append(f"Loader workers: {args.loader_workers}")
    lines.append(f"Total wall time (s): {total_wall_time:.2f}")
    lines.append("")
    lines.append("LAST ROUND")
    lines.append("-" * 72)
    lines.append(f"Round: {last['round']}")
    lines.append(f"Train loss: {last['train_loss']:.4f}")
    lines.append(f"Train acc: {last['train_acc']:.4f}")
    lines.append(f"Train top5: {last['train_top5']:.4f}")
    lines.append(f"Val loss: {last['val_loss']:.4f}")
    lines.append(f"Val acc: {last['val_acc']:.4f}")
    lines.append(f"Val top5: {last['val_top5']:.4f}")
    lines.append("")
    lines.append("BEST")
    lines.append("-" * 72)
    lines.append(f"Best val_acc: round {best_acc['round']} -> {best_acc['val_acc']:.4f}")
    lines.append(f"Best val_top5: round {best_top5['round']} -> {best_top5['val_top5']:.4f}")
    lines.append(f"Best val_loss: round {best_loss['round']} -> {best_loss['val_loss']:.4f}")
    lines.append("")
    lines.append("PROGRESS")
    lines.append("-" * 72)
    lines.append(f"Delta val_acc:  {last['val_acc'] - first['val_acc']:+.4f}")
    lines.append(f"Delta val_top5: {last['val_top5'] - first['val_top5']:+.4f}")
    lines.append(f"Delta val_loss: {first['val_loss'] - last['val_loss']:+.4f}")

    with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_final_analysis(history, run_dir):
    if not history:
        log("No history available for final analysis", "WARN")
        return

    first = history[0]
    last = history[-1]
    best_acc = max(history, key=lambda x: x["val_acc"])
    best_top5 = max(history, key=lambda x: x["val_top5"])
    best_loss = min(history, key=lambda x: x["val_loss"])

    print()
    print(color("=" * 88, C.MAGENTA))
    print(color(" FINAL ANALYSIS ".center(88), C.BOLD + C.GREEN))
    print(color("=" * 88, C.MAGENTA))
    print(f"Run directory           : {run_dir}")
    print(f"Rounds completed        : {len(history)}")
    print(f"Final train loss        : {last['train_loss']:.4f}")
    print(f"Final train acc         : {last['train_acc']:.4f}")
    print(f"Final train top5        : {last['train_top5']:.4f}")
    print(f"Final validation loss   : {last['val_loss']:.4f}")
    print(f"Final validation acc    : {last['val_acc']:.4f}")
    print(f"Final validation top5   : {last['val_top5']:.4f}")
    print("-" * 88)
    print(f"Best validation acc     : round {best_acc['round']} -> {best_acc['val_acc']:.4f}")
    print(f"Best validation top5    : round {best_top5['round']} -> {best_top5['val_top5']:.4f}")
    print(f"Best validation loss    : round {best_loss['round']} -> {best_loss['val_loss']:.4f}")
    print("-" * 88)
    print(f"Delta validation acc    : {last['val_acc'] - first['val_acc']:+.4f}")
    print(f"Delta validation top5   : {last['val_top5'] - first['val_top5']:+.4f}")
    print(f"Delta validation loss   : {first['val_loss'] - last['val_loss']:+.4f}")
    print("-" * 88)
    print("Artifacts generated:")
    print(f"  • {run_dir / 'metrics.csv'}")
    print(f"  • {run_dir / 'metrics.json'}")
    print(f"  • {run_dir / 'summary.txt'}")
    print(f"  • {run_dir / 'loss_curve.png'}")
    print(f"  • {run_dir / 'accuracy_top1_curve.png'}")
    print(f"  • {run_dir / 'accuracy_top5_curve.png'}")
    print(f"  • {run_dir / 'throughput_curve.png'}")
    print(f"  • {run_dir / 'round_time_curve.png'}")
    print(f"  • {run_dir / 'federated_report.ipynb'}")
    print(color("=" * 88, C.MAGENTA), flush=True)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--socket-timeout", type=float, default=7200.0)
    parser.add_argument("--prefer-mps", action="store_true")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--checkpoint-every-round", action="store_true")
    args = parser.parse_args()

    wall_start = time.time()
    banner()

    run_dir = make_run_dir(args.output_dir)
    device = select_device(prefer_mps=args.prefer_mps)
    args.device_name = str(device)

    log(f"Run directory: {run_dir}", "OK")
    log(f"Selected device: {device}", "OK")
    log("Loading validation dataset...", "STEP")
    val_loader = get_val_loader(args.data_dir, args.val_batch_size, args.loader_workers)
    log(f"Validation batches: {len(val_loader)}", "OK")

    log("Building global model...", "STEP")
    global_model = build_model(num_classes=200).to(device)
    num_params = sum(p.numel() for p in global_model.parameters())
    log(f"Model parameters: {num_params:,}", "OK")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(args.num_workers)

    log(f"Listening on {args.host}:{args.port}", "OK")
    log(f"Waiting for {args.num_workers} workers...", "STEP")

    workers = []
    history = []

    for worker_id in range(args.num_workers):
        conn, addr = server_sock.accept()
        conn.settimeout(args.socket_timeout)

        try:
            hello = recv_msg(conn)
            worker_name = hello.get("worker_name", f"worker-{worker_id}")
        except Exception:
            worker_name = f"worker-{worker_id}"

        worker = {
            "id": worker_id,
            "name": worker_name,
            "addr": addr,
            "conn": conn,
            "alive": True,
        }
        workers.append(worker)
        log(
            f"Worker connected | id={worker_id} | name={worker_name} | addr={addr[0]}:{addr[1]}",
            "OK",
        )

    try:
        for round_idx in range(1, args.rounds + 1):
            alive_workers = [w for w in workers if w["alive"]]
            if not alive_workers:
                log("No active workers remain. Aborting training.", "ERR")
                break

            print()
            print(color("─" * 88, C.BLUE))
            log(f"Starting round {round_idx}/{args.rounds}", "STEP")
            round_start = time.time()

            log("Serializing global model state...", "STEP")
            global_state = state_dict_to_cpu(global_model.state_dict())

            sent_workers = []
            log(f"Dispatching round {round_idx} to {len(alive_workers)} active workers...", "STEP")

            for worker in alive_workers:
                msg = {
                    "type": "train",
                    "round": round_idx,
                    "worker_id": worker["id"],
                    "partition_count": args.num_workers,
                    "state_dict": global_state,
                    "local_epochs": args.local_epochs,
                    "lr": args.lr,
                    "batch_size": args.train_batch_size,
                    "loader_workers": args.loader_workers,
                }

                try:
                    send_msg(worker["conn"], msg)
                    sent_workers.append(worker)
                    log(
                        f"Round {round_idx} sent to worker {worker['id']} ({worker['name']})",
                        "OK",
                    )
                except Exception as e:
                    worker["alive"] = False
                    log(
                        f"Failed sending round {round_idx} to worker {worker['id']} ({worker['name']}): {repr(e)}",
                        "ERR",
                    )

            results = []
            log("Waiting for worker updates...", "STEP")

            for worker in sent_workers:
                if not worker["alive"]:
                    continue

                try:
                    reply = recv_msg(worker["conn"])
                except Exception as e:
                    worker["alive"] = False
                    log(
                        f"Worker {worker['id']} ({worker['name']}) disconnected during round {round_idx}: {repr(e)}",
                        "ERR",
                    )
                    continue

                msg_type = reply.get("type", "unknown")

                if msg_type == "result":
                    results.append(reply)
                    log(
                        f"Worker {reply['worker_id']} done | "
                        f"loss={reply['train_loss']:.4f} | "
                        f"acc={reply['train_acc']:.4f} | "
                        f"top5={reply['train_top5']:.4f} | "
                        f"samples={reply['num_samples']} | "
                        f"time={reply['train_time']:.2f}s",
                        "METRIC",
                    )
                elif msg_type == "error":
                    worker["alive"] = False
                    log(
                        f"Worker {worker['id']} reported error: {reply.get('error', 'unknown')}",
                        "ERR",
                    )
                    tb = reply.get("traceback", "")
                    if tb:
                        print(tb, flush=True)
                else:
                    worker["alive"] = False
                    log(
                        f"Worker {worker['id']} responded with invalid message type: {msg_type}",
                        "ERR",
                    )

            if not results:
                log("No valid worker updates received in this round.", "ERR")
                break

            log("Aggregating worker models (weighted average)...", "STEP")
            new_state = weighted_average_state_dict(results)
            global_model.load_state_dict(new_state, strict=True)

            if args.checkpoint_every_round:
                ckpt_path = run_dir / f"global_model_round_{round_idx:03d}.pth"
                torch.save(global_model.state_dict(), ckpt_path)
                log(f"Checkpoint saved: {ckpt_path.name}", "OK")

            log("Running validation on aggregated global model...", "STEP")
            val_metrics = evaluate(global_model, val_loader, device)

            round_time = time.time() - round_start
            total_samples = sum(r["num_samples"] for r in results)
            avg_train_loss = sum(r["train_loss"] * r["num_samples"] for r in results) / total_samples
            avg_train_acc = sum(r["train_acc"] * r["num_samples"] for r in results) / total_samples
            avg_train_top5 = sum(r["train_top5"] * r["num_samples"] for r in results) / total_samples
            throughput = total_samples / round_time

            round_record = {
                "round": round_idx,
                "train_loss": float(avg_train_loss),
                "train_acc": float(avg_train_acc),
                "train_top5": float(avg_train_top5),
                "val_loss": float(val_metrics["loss"]),
                "val_acc": float(val_metrics["top1"]),
                "val_top5": float(val_metrics["top5"]),
                "round_time": float(round_time),
                "throughput": float(throughput),
                "num_samples": int(total_samples),
                "active_workers": int(len(results)),
            }
            history.append(round_record)

            print(color("┌" + "─" * 86 + "┐", C.GREEN))
            print(color(
                f"│ ROUND {round_idx:02d} SUMMARY".ljust(87) + "│",
                C.GREEN,
            ))
            print(color("├" + "─" * 86 + "┤", C.GREEN))
            print(f"│ train_loss={avg_train_loss:.4f} | train_acc={avg_train_acc:.4f} | train_top5={avg_train_top5:.4f}".ljust(87) + "│")
            print(f"│ val_loss  ={val_metrics['loss']:.4f} | val_acc  ={val_metrics['top1']:.4f} | val_top5  ={val_metrics['top5']:.4f}".ljust(87) + "│")
            print(f"│ round_time={round_time:.2f}s | throughput={throughput:.2f} samples/s | active_workers={len(results)}".ljust(87) + "│")
            print(color("└" + "─" * 86 + "┘", C.GREEN), flush=True)

        if history:
            log("Saving metrics to CSV and JSON...", "STEP")
            save_metrics_csv(history, run_dir / "metrics.csv")
            save_metrics_json(history, run_dir / "metrics.json")

            log("Generating plots...", "STEP")
            save_all_plots(history, run_dir)

            log("Generating notebook report...", "STEP")
            create_notebook(run_dir)

            log("Writing textual summary...", "STEP")
            write_summary(history, run_dir, args, time.time() - wall_start)

            if args.save_model:
                model_path = run_dir / "global_model_final.pth"
                torch.save(global_model.state_dict(), model_path)
                log(f"Final model saved: {model_path.name}", "OK")

            print_final_analysis(history, run_dir)

    finally:
        log("Sending shutdown signal to workers...", "STEP")
        for worker in workers:
            try:
                if worker["alive"]:
                    send_msg(worker["conn"], {"type": "shutdown"})
            except Exception:
                pass
            try:
                worker["conn"].close()
            except Exception:
                pass

        try:
            server_sock.close()
        except Exception:
            pass

        log("Server closed cleanly.", "OK")


if __name__ == "__main__":
    main()