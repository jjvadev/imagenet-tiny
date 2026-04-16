from __future__ import annotations

import argparse
import gc
import socket
import traceback

import torch

from common import (
    build_model,
    make_partitioned_train_loader,
    recv_msg,
    seed_everything,
    select_device,
    send_msg,
    state_dict_to_cpu,
    train_one_round,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5000)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--name", type=str, default="worker")
    parser.add_argument("--socket-timeout", type=float, default=7200.0)
    parser.add_argument("--prefer-mps", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = select_device(prefer_mps=args.prefer_mps)

    print(f"[WORKER {args.name}] Device: {device}")
    print(f"[WORKER {args.name}] Conectando a {args.server_host}:{args.server_port}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(args.socket_timeout)
    sock.connect((args.server_host, args.server_port))
    send_msg(sock, {"type": "hello", "worker_name": args.name})

    model = None
    current_arch = None

    cached_loader = None
    cached_num_samples = None
    cached_partition = None
    cached_batch_size = None
    cached_loader_workers = None
    cached_image_size = None

    try:
        while True:
            msg = recv_msg(sock)
            msg_type = msg.get("type")

            if msg_type == "shutdown":
                print(f"[WORKER {args.name}] Shutdown recibido.")
                break

            if msg_type != "train":
                raise ValueError(f"Mensaje no soportado: {msg_type}")

            epoch = msg["epoch"]
            worker_id = msg["worker_id"]
            partition_count = msg["partition_count"]
            lr = msg["lr"]
            optimizer = msg["optimizer"]
            weight_decay = msg["weight_decay"]
            momentum = msg["momentum"]
            local_epochs = msg["local_epochs"]
            batch_size = msg["batch_size"]
            loader_workers = msg["loader_workers"]
            image_size = msg["image_size"]
            arch = msg["arch"]

            print(f"\n[WORKER {args.name}] ===== Epoch {epoch} =====")

            if model is None or current_arch != arch:
                model = build_model(arch=arch, num_classes=200).to(device)
                current_arch = arch
                print(f"[WORKER {args.name}] Modelo creado: {arch}")

            if (
                cached_loader is None
                or cached_partition != (worker_id, partition_count)
                or cached_batch_size != batch_size
                or cached_loader_workers != loader_workers
                or cached_image_size != image_size
            ):
                cached_loader, cached_num_samples = make_partitioned_train_loader(
                    data_dir=args.data_dir,
                    worker_id=worker_id,
                    partition_count=partition_count,
                    batch_size=batch_size,
                    loader_workers=loader_workers,
                    image_size=image_size,
                    seed=args.seed,
                )
                cached_partition = (worker_id, partition_count)
                cached_batch_size = batch_size
                cached_loader_workers = loader_workers
                cached_image_size = image_size

                print(
                    f"[WORKER {args.name}] Particion lista | "
                    f"samples={cached_num_samples} | batches={len(cached_loader)}"
                )

            model.load_state_dict(msg["state_dict"], strict=True)
            model.to(device)

            metrics = train_one_round(
                model=model,
                loader=cached_loader,
                device=device,
                optimizer_name=optimizer,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                local_epochs=local_epochs,
                use_amp=True,
            )

            reply = {
                "type": "result",
                "epoch": epoch,
                "worker_id": worker_id,
                "state_dict": state_dict_to_cpu(model.state_dict()),
                "train_loss": float(metrics["train_loss"]),
                "train_acc": float(metrics["train_acc"]),
                "num_samples": int(metrics["num_samples"]),
                "train_time": float(metrics["train_time"]),
            }

            print(f"[WORKER {args.name}] reply keys: {list(reply.keys())}")

            send_msg(sock, reply)

            print(
                f"[WORKER {args.name}] "
                f"type={reply['type']} | "
                f"train_loss={reply['train_loss']:.6f} | "
                f"train_acc={reply['train_acc']:.6f} | "
                f"samples={reply['num_samples']} | "
                f"time={reply['train_time']:.2f}s"
            )

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[WORKER {args.name}] ERROR: {e}")
        print(tb)
        try:
            send_msg(
                sock,
                {
                    "type": "error",
                    "worker_name": args.name,
                    "error": str(e),
                    "traceback": tb,
                },
            )
        except Exception:
            pass
    finally:
        try:
            sock.close()
        except Exception:
            pass
        print(f"[WORKER {args.name}] Cerrado correctamente.")


if __name__ == "__main__":
    main()