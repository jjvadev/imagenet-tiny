import argparse
import os
import pickle
import socket
import struct
import time
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ----------------------------
# Socket helpers
# ----------------------------
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


# ----------------------------
# Model / metrics
# ----------------------------
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

        if torch.is_floating_point(ref_tensor):
            acc = torch.zeros_like(ref_tensor, dtype=torch.float32)
            for payload in worker_payloads:
                weight = payload["num_samples"] / total_samples
                acc += payload["state_dict"][key].float() * weight
            agg[key] = acc.to(ref_tensor.dtype)
        else:
            # Ej. num_batches_tracked
            agg[key] = ref_tensor.clone()

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


# ----------------------------
# Tiny-ImageNet dataset
# ----------------------------
class TinyImageNetDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        wnids_path = self.root / "wnids.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(f"No existe: {wnids_path}")

        with open(wnids_path, "r") as f:
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
            with open(ann_file, "r") as f:
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

    for images, targets in loader:
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

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
        "samples": total_samples,
    }


# ----------------------------
# Main server
# ----------------------------
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
    parser.add_argument("--loader-workers", type=int, default=2)
    parser.add_argument("--socket-timeout", type=float, default=7200.0)
    parser.add_argument("--prefer-mps", action="store_true")
    args = parser.parse_args()

    device = select_device(prefer_mps=args.prefer_mps)
    print(f"[Server] device={device}")

    val_loader = get_val_loader(
        data_dir=args.data_dir,
        batch_size=args.val_batch_size,
        num_workers=args.loader_workers,
    )

    global_model = build_model(num_classes=200).to(device)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(args.num_workers)

    print(f"[Server] listening on {args.host}:{args.port}")
    print(f"[Server] waiting for {args.num_workers} workers...")

    workers = []

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
        print(f"[Server] connected: id={worker_id} name={worker_name} addr={addr}")

    try:
        for round_idx in range(1, args.rounds + 1):
            alive_workers = [w for w in workers if w["alive"]]
            if not alive_workers:
                print("[Server] no quedan workers vivos, abortando")
                break

            print(f"\n[Server] Round {round_idx}/{args.rounds}")
            round_start = time.time()

            global_state = state_dict_to_cpu(global_model.state_dict())
            sent_workers = []

            for worker in alive_workers:
                msg = {
                    "type": "train",
                    "round": round_idx,
                    "worker_id": worker["id"],
                    "partition_count": args.num_workers,   # fijo para no cambiar particiones
                    "state_dict": global_state,
                    "local_epochs": args.local_epochs,
                    "lr": args.lr,
                    "batch_size": args.train_batch_size,
                    "loader_workers": args.loader_workers,
                }

                try:
                    send_msg(worker["conn"], msg)
                    sent_workers.append(worker)
                except Exception as e:
                    worker["alive"] = False
                    print(
                        f"[Server] Worker {worker['id']} ({worker['name']}) "
                        f"falló al enviar ronda {round_idx}: {repr(e)}"
                    )

            results = []

            for worker in sent_workers:
                if not worker["alive"]:
                    continue

                try:
                    reply = recv_msg(worker["conn"])
                except Exception as e:
                    worker["alive"] = False
                    print(
                        f"[Server] Worker {worker['id']} ({worker['name']}) "
                        f"se desconectó en ronda {round_idx}: {repr(e)}"
                    )
                    continue

                msg_type = reply.get("type", "unknown")

                if msg_type == "result":
                    results.append(reply)
                    print(
                        f"  Worker {reply['worker_id']}: "
                        f"loss={reply['train_loss']:.4f}, "
                        f"top1={reply['train_top1']:.4f}, "
                        f"top5={reply['train_top5']:.4f}, "
                        f"samples={reply['num_samples']}, "
                        f"time={reply['train_time']:.2f}s"
                    )

                elif msg_type == "error":
                    worker["alive"] = False
                    print(
                        f"  Worker {worker['id']} reportó error: "
                        f"{reply.get('error', 'unknown')}"
                    )

                else:
                    worker["alive"] = False
                    print(
                        f"  Worker {worker['id']} respondió algo inválido: {msg_type}"
                    )

            if not results:
                print("[Server] no hubo actualizaciones válidas en esta ronda")
                break

            new_state = weighted_average_state_dict(results)
            global_model.load_state_dict(new_state, strict=True)

            val_metrics = evaluate(global_model, val_loader, device)

            round_time = time.time() - round_start
            total_samples = sum(r["num_samples"] for r in results)

            avg_train_loss = sum(r["train_loss"] * r["num_samples"] for r in results) / total_samples
            avg_train_top1 = sum(r["train_top1"] * r["num_samples"] for r in results) / total_samples
            avg_train_top5 = sum(r["train_top5"] * r["num_samples"] for r in results) / total_samples

            throughput = total_samples / round_time

            print(
                f"[Server] Round {round_idx} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"train_top1={avg_train_top1:.4f} | "
                f"train_top5={avg_train_top5:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_top1={val_metrics['top1']:.4f} | "
                f"val_top5={val_metrics['top5']:.4f} | "
                f"time={round_time:.2f}s | "
                f"throughput={throughput:.2f} samples/s"
            )

    finally:
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

        print("[Server] cerrado")


if __name__ == "__main__":
    main()