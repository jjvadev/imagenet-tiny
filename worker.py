import argparse
import gc
import math
import pickle
import socket
import struct
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
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


def banner(name):
    line = "=" * 88
    print(color(line, C.BLUE))
    print(color(f" FEDERATED WORKER [{name}] ".center(88), C.BOLD + C.GREEN))
    print(color(line, C.BLUE), flush=True)


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
        else:
            raise ValueError("This worker only supports split='train'")

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


def make_partitioned_loader(data_dir, worker_id, partition_count, batch_size, loader_workers):
    normalize = transforms.Normalize(
        mean=[0.4802, 0.4481, 0.3975],
        std=[0.2302, 0.2265, 0.2262],
    )

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = TinyImageNetDataset(data_dir, split="train", transform=transform)

    g = torch.Generator()
    g.manual_seed(12345)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    indices = perm[worker_id::partition_count]

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=False,
    )
    return loader, len(subset)


def train_one_round(
    model,
    loader,
    device,
    lr,
    local_epochs,
    worker_name="worker",
    log_interval=50,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_top5 = 0.0
    total_samples = 0

    t0 = time.time()
    total_batches = len(loader) * local_epochs
    global_batch = 0

    log(
        f"[{worker_name}] Local training starting | epochs={local_epochs} | batches/epoch={len(loader)} | lr={lr}",
        "STEP",
    )

    for epoch in range(local_epochs):
        log(f"[{worker_name}] Epoch {epoch + 1}/{local_epochs} started", "STEP")

        for batch_idx, (images, targets) in enumerate(loader, start=1):
            global_batch += 1

            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            bsz = targets.size(0)
            acc, top5 = topk_accuracy(logits, targets, topk=(1, 5))

            total_loss += loss.item() * bsz
            total_acc += acc
            total_top5 += top5
            total_samples += bsz

            if batch_idx % log_interval == 0 or batch_idx == len(loader):
                avg_loss = total_loss / total_samples
                avg_acc = total_acc / total_samples
                avg_top5 = total_top5 / total_samples
                progress = (global_batch / total_batches) * 100.0

                log(
                    f"[{worker_name}] progress={progress:6.2f}% | "
                    f"epoch={epoch + 1}/{local_epochs} | "
                    f"batch={batch_idx}/{len(loader)} | "
                    f"avg_loss={avg_loss:.4f} | "
                    f"avg_acc={avg_acc:.4f} | "
                    f"avg_top5={avg_top5:.4f}",
                    "METRIC",
                )

    elapsed = time.time() - t0
    return {
        "train_loss": total_loss / total_samples,
        "train_acc": total_acc / total_samples,
        "train_top5": total_top5 / total_samples,
        "num_samples": total_samples,
        "train_time": elapsed,
    }


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5000)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--name", type=str, default="worker")
    parser.add_argument("--socket-timeout", type=float, default=7200.0)
    parser.add_argument("--prefer-mps", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    banner(args.name)

    device = select_device(prefer_mps=args.prefer_mps)
    log(f"Worker name: {args.name}", "OK")
    log(f"Selected device: {device}", "OK")
    log(f"Dataset root: {args.data_dir}", "OK")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(args.socket_timeout)

    log(f"Connecting to server {args.server_host}:{args.server_port}...", "STEP")
    sock.connect((args.server_host, args.server_port))
    send_msg(sock, {"type": "hello", "worker_name": args.name})
    log("Handshake sent successfully", "OK")

    model = build_model(num_classes=200).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {num_params:,}", "OK")

    cached_loader = None
    cached_num_samples = None
    cached_partition = None
    cached_batch_size = None
    cached_loader_workers = None

    try:
        while True:
            log("Waiting for server message...", "STEP")
            msg = recv_msg(sock)
            msg_type = msg.get("type")

            if msg_type == "shutdown":
                log("Shutdown signal received. Worker exiting.", "WARN")
                break

            if msg_type != "train":
                raise ValueError(f"Unsupported message type: {msg_type}")

            round_idx = msg["round"]
            worker_id = msg["worker_id"]
            partition_count = msg["partition_count"]
            lr = msg["lr"]
            local_epochs = msg["local_epochs"]
            batch_size = msg["batch_size"]
            loader_workers = msg["loader_workers"]

            print()
            print(color("─" * 88, C.CYAN))
            log(
                f"Round {round_idx} received | worker_id={worker_id} | partition_count={partition_count}",
                "STEP",
            )

            if (
                cached_loader is None
                or cached_partition != (worker_id, partition_count)
                or cached_batch_size != batch_size
                or cached_loader_workers != loader_workers
            ):
                log("Preparing local data partition...", "STEP")
                cached_loader, cached_num_samples = make_partitioned_loader(
                    data_dir=args.data_dir,
                    worker_id=worker_id,
                    partition_count=partition_count,
                    batch_size=batch_size,
                    loader_workers=loader_workers,
                )
                cached_partition = (worker_id, partition_count)
                cached_batch_size = batch_size
                cached_loader_workers = loader_workers

                log(
                    f"Local partition ready | samples={cached_num_samples} | batches={len(cached_loader)}",
                    "OK",
                )
            else:
                log("Reusing cached local dataloader", "OK")

            log("Loading global weights from server...", "STEP")
            model.load_state_dict(msg["state_dict"], strict=True)
            model.to(device)

            metrics = train_one_round(
                model=model,
                loader=cached_loader,
                device=device,
                lr=lr,
                local_epochs=local_epochs,
                worker_name=args.name,
                log_interval=args.log_interval,
            )

            reply = {
                "type": "result",
                "round": round_idx,
                "worker_id": worker_id,
                "state_dict": state_dict_to_cpu(model.state_dict()),
                "train_loss": metrics["train_loss"],
                "train_acc": metrics["train_acc"],
                "train_top5": metrics["train_top5"],
                "num_samples": metrics["num_samples"],
                "train_time": metrics["train_time"],
            }

            log("Sending updated model back to server...", "STEP")
            send_msg(sock, reply)

            print(color("┌" + "─" * 86 + "┐", C.GREEN))
            print(color(
                f"│ WORKER {args.name} ROUND {round_idx:02d} COMPLETED".ljust(87) + "│",
                C.GREEN,
            ))
            print(color("├" + "─" * 86 + "┤", C.GREEN))
            print(f"│ train_loss={metrics['train_loss']:.4f} | train_acc={metrics['train_acc']:.4f} | train_top5={metrics['train_top5']:.4f}".ljust(87) + "│")
            print(f"│ num_samples={metrics['num_samples']} | train_time={metrics['train_time']:.2f}s".ljust(87) + "│")
            print(color("└" + "─" * 86 + "┘", C.GREEN), flush=True)

            gc.collect()
            if device.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    except Exception as e:
        tb = traceback.format_exc()
        log(f"Fatal worker error: {e}", "ERR")
        print(tb, flush=True)
        try:
            send_msg(sock, {
                "type": "error",
                "worker_name": args.name,
                "error": str(e),
                "traceback": tb,
            })
        except Exception:
            pass
    finally:
        try:
            sock.close()
        except Exception:
            pass
        log("Worker closed cleanly.", "OK")


if __name__ == "__main__":
    main()