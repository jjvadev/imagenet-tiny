from __future__ import annotations

import argparse
import csv
import json
import socket
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from analysis_notebook import generate_analysis_notebook
from common import (
    build_model,
    evaluate,
    make_val_loader,
    recv_msg,
    seed_everything,
    select_device,
    send_msg,
    state_dict_to_cpu,
    weighted_average_state_dict,
)


def make_run_dir(base_dir="runs"):
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{ts_now}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_history_csv(history, out_path: Path):
    fieldnames = ["epoch", "cost", "train", "test", "epoch_time", "total_time", "lr"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row[k] for k in fieldnames})


def save_summary_json(summary: dict, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def save_plots(history, run_dir: Path):
    epochs = [h["epoch"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["cost"] for h in history], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Cost por epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "cost_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train"] for h in history], marker="o", label="Train")
    plt.plot(epochs, [h["test"] for h in history], marker="s", label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy por epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "accuracy_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["epoch_time"] for h in history], marker="o", label="Epoch")
    plt.plot(epochs, [h["total_time"] for h in history], marker="s", label="Acumulado")
    plt.xlabel("Epoch")
    plt.ylabel("Segundos")
    plt.title("Tiempo de entrenamiento")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "time_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["lr"] for h in history], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("LR por epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "lr_curve.png")
    plt.close()


def compute_epoch_lr(base_lr: float, epoch: int, step_size: int, gamma: float) -> float:
    if step_size <= 0:
        return base_lr
    factor = epoch // step_size
    return base_lr * (gamma ** factor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-step-size", type=int, default=20)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "sgd", "adamw"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--arch", type=str, default="resnet18", choices=["small_cnn", "resnet18"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--max-val-samples", type=int, default=2000)
    parser.add_argument("--socket-timeout", type=float, default=7200.0)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--prefer-mps", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--save-best-model", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = select_device(prefer_mps=args.prefer_mps)
    run_dir = make_run_dir(args.output_dir)
    wall_start = None

    print(f"[SERVER] Run dir: {run_dir}")
    print(f"[SERVER] Device: {device}")
    print(f"[SERVER] Cargando validacion...")

    val_loader = make_val_loader(
        data_dir=args.data_dir,
        batch_size=args.val_batch_size,
        loader_workers=args.loader_workers,
        image_size=args.image_size,
        max_samples=args.max_val_samples,
        seed=args.seed,
        arch=args.arch,
        pretrained=args.pretrained,
    )

    print("[SERVER] Construyendo modelo global...")
    global_model = build_model(
        arch=args.arch,
        num_classes=200,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(args.num_workers)

    print(f"[SERVER] Escuchando en {args.host}:{args.port}")
    print(f"[SERVER] Esperando {args.num_workers} workers...")

    workers = []
    history = []
    best_test = -1.0
    best_epoch = -1

    for worker_id in range(args.num_workers):
        conn, addr = server_sock.accept()
        conn.settimeout(args.socket_timeout)
        hello = recv_msg(conn)
        worker_name = hello.get("worker_name", f"worker-{worker_id}")
        workers.append(
            {
                "id": worker_id,
                "name": worker_name,
                "conn": conn,
                "addr": addr,
                "alive": True,
            }
        )
        print(f"[SERVER] Conectado worker {worker_id}: {worker_name} @ {addr[0]}:{addr[1]}")

    try:
        for epoch in range(args.rounds):
            if wall_start is None:
                wall_start = time.time()

            epoch_start = time.time()
            active_workers = [w for w in workers if w["alive"]]

            if not active_workers:
                print("[SERVER] No hay workers activos.")
                break

            current_lr = compute_epoch_lr(args.lr, epoch, args.lr_step_size, args.lr_gamma)

            print(f"\n[SERVER] ===== Epoch {epoch} / {args.rounds - 1} =====")
            print(f"[SERVER] LR actual: {current_lr:.8f}")

            global_state = state_dict_to_cpu(global_model.state_dict())
            sent_workers = []

            for worker in active_workers:
                msg = {
                    "type": "train",
                    "epoch": epoch,
                    "worker_id": worker["id"],
                    "partition_count": args.num_workers,
                    "state_dict": global_state,
                    "local_epochs": args.local_epochs,
                    "lr": current_lr,
                    "optimizer": args.optimizer,
                    "weight_decay": args.weight_decay,
                    "momentum": args.momentum,
                    "batch_size": args.train_batch_size,
                    "loader_workers": args.loader_workers,
                    "image_size": args.image_size,
                    "arch": args.arch,
                    "pretrained": args.pretrained,
                    "freeze_backbone": args.freeze_backbone,
                }

                try:
                    send_msg(worker["conn"], msg)
                    sent_workers.append(worker)
                except Exception as e:
                    worker["alive"] = False
                    print(f"[SERVER] Error enviando a {worker['name']}: {e}")

            results = []

            for worker in sent_workers:
                try:
                    reply = recv_msg(worker["conn"])
                except Exception as e:
                    worker["alive"] = False
                    print(f"[SERVER] Worker caido {worker['name']}: {e}")
                    continue

                if reply.get("type") == "result":
                    results.append(reply)
                    print(
                        f"[SERVER] {worker['name']} | "
                        f"train_loss={reply['train_loss']:.6f} | "
                        f"train_acc={reply['train_acc']:.6f} | "
                        f"samples={reply['num_samples']} | "
                        f"time={reply['train_time']:.2f}s"
                    )
                else:
                    worker["alive"] = False
                    print(f"[SERVER] Respuesta invalida de {worker['name']}: {reply.get('type')}")

            if not results:
                print("[SERVER] No hubo resultados validos.")
                break

            new_state = weighted_average_state_dict(results)
            global_model.load_state_dict(new_state, strict=True)

            metrics = evaluate(global_model, val_loader, device=device, use_amp=True)

            total_samples = sum(r["num_samples"] for r in results)
            train_acc = sum(r["train_acc"] * r["num_samples"] for r in results) / total_samples

            epoch_time = time.time() - epoch_start
            total_time = time.time() - wall_start

            row = {
                "epoch": int(epoch),
                "cost": float(metrics["loss"]),
                "train": float(train_acc),
                "test": float(metrics["acc"]),
                "epoch_time": float(epoch_time),
                "total_time": float(total_time),
                "lr": float(current_lr),
            }
            history.append(row)

            if row["test"] > best_test:
                best_test = row["test"]
                best_epoch = row["epoch"]
                if args.save_best_model:
                    torch.save(global_model.state_dict(), run_dir / "model_best.pth")
                    print(f"[SERVER] Nuevo mejor modelo guardado en epoch {best_epoch} con test={best_test:.6f}")

            print(
                f"[SERVER] epoch={row['epoch']} | "
                f"cost={row['cost']:.6f} | "
                f"train={row['train']:.6f} | "
                f"test={row['test']:.6f} | "
                f"epoch_time={row['epoch_time']:.2f}s | "
                f"total_time={row['total_time']:.2f}s"
            )

        if history:
            history_csv = run_dir / "history.csv"
            summary_json = run_dir / "summary.json"

            save_history_csv(history, history_csv)

            summary = {
                "arch": args.arch,
                "optimizer": args.optimizer,
                "epochs_completed": len(history),
                "num_workers": args.num_workers,
                "local_epochs": args.local_epochs,
                "lr": args.lr,
                "lr_step_size": args.lr_step_size,
                "lr_gamma": args.lr_gamma,
                "train_batch_size": args.train_batch_size,
                "val_batch_size": args.val_batch_size,
                "device": str(device),
                "best_test": float(best_test),
                "best_epoch": int(best_epoch),
                "total_time_sec": float(history[-1]["total_time"]),
                "pretrained": bool(args.pretrained),
                "freeze_backbone": bool(args.freeze_backbone),
                "image_size": int(args.image_size),
            }
            save_summary_json(summary, summary_json)

            save_plots(history, run_dir)

            generate_analysis_notebook(
                results_dir=str(run_dir),
                summary=summary,
                history_csv_path=str(history_csv),
                summary_json_path=str(summary_json),
            )

            if args.save_model:
                torch.save(global_model.state_dict(), run_dir / "model_final.pth")

            print(f"\n[SERVER] Historial guardado en: {history_csv}")
            print(f"[SERVER] Summary guardado en: {summary_json}")
            print(f"[SERVER] Carpeta final: {run_dir}")

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

        print("[SERVER] Cerrado correctamente.")


if __name__ == "__main__":
    main()