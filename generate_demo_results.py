import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

from analysis_notebook import generate_analysis_notebook

OUT_DIR = Path("/mnt/data/imagenet_project/sockets_v2-imagenet/results/imagenet_demo_resnet18_pretrained_frozen")
OUT_DIR.mkdir(parents=True, exist_ok=True)

history = []
train_loss = [3.80, 3.35, 3.01, 2.79, 2.61, 2.48, 2.36, 2.29, 2.22, 2.16, 2.11, 2.07]
val_loss   = [3.95, 3.55, 3.22, 3.01, 2.86, 2.75, 2.66, 2.61, 2.58, 2.55, 2.53, 2.51]
train_top1 = [0.28, 0.34, 0.40, 0.45, 0.49, 0.52, 0.55, 0.57, 0.59, 0.60, 0.61, 0.62]
val_top1   = [0.24, 0.30, 0.36, 0.41, 0.45, 0.48, 0.51, 0.53, 0.55, 0.56, 0.57, 0.58]
train_top5 = [0.50, 0.59, 0.66, 0.72, 0.76, 0.79, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88]
val_top5   = [0.46, 0.55, 0.63, 0.69, 0.73, 0.76, 0.79, 0.81, 0.82, 0.83, 0.84, 0.85]
epoch_time = [92, 90, 89, 88, 88, 87, 87, 87, 86, 86, 86, 85]

total = 0
for i in range(len(train_loss)):
    total += epoch_time[i]
    history.append({
        "epoch": i,
        "train_cost": train_loss[i],
        "train_top1": train_top1[i],
        "train_top5": train_top5[i],
        "val_cost": val_loss[i],
        "val_top1": val_top1[i],
        "val_top5": val_top5[i],
        "worker_cost_mean": train_loss[i] + 0.05,
        "worker_top1_mean": max(train_top1[i] - 0.02, 0),
        "worker_top5_mean": max(train_top5[i] - 0.03, 0),
        "epoch_time": epoch_time[i],
        "total_time": total,
    })

summary = {
    "run_name": "imagenet_demo_resnet18_pretrained_frozen",
    "dataset": "ImageNet",
    "device": "cpu (demo)",
    "workers": 2,
    "epochs": len(history),
    "local_epochs": 1,
    "batch_size": 32,
    "eval_batch_size": 64,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "optimizer": "sgd",
    "momentum": 0.9,
    "arch": "resnet18",
    "pretrained": True,
    "freeze_backbone": True,
    "dropout": 0.2,
    "image_size": 224,
    "seed": 42,
    "augmentation": True,
    "train_samples": 20000,
    "val_samples": 5000,
    "loader_workers": 4,
    "final_train_loss": history[-1]["train_cost"],
    "final_train_top1": history[-1]["train_top1"],
    "final_train_top5": history[-1]["train_top5"],
    "final_val_loss": history[-1]["val_cost"],
    "final_val_top1": history[-1]["val_top1"],
    "final_val_top5": history[-1]["val_top5"],
    "best_val_top1": max(h["val_top1"] for h in history),
    "best_round": max(range(len(history)), key=lambda i: history[i]["val_top1"]),
    "total_time_sec": history[-1]["total_time"],
    "history_points": len(history),
    "demo_only": True,
    "note": "Resultados ilustrativos para mostrar el formato de salida, las gráficas y el notebook.",
}

csv_path = OUT_DIR / "history.csv"
json_path = OUT_DIR / "summary.json"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
    writer.writeheader()
    writer.writerows(history)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

rounds = [h["epoch"] for h in history]

plt.figure(figsize=(8, 5))
plt.plot(rounds, [h["train_cost"] for h in history], marker="o", label="Loss train")
plt.plot(rounds, [h["val_cost"] for h in history], marker="s", label="Loss val")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("Loss por round")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "loss_curve.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(rounds, [h["train_top1"] for h in history], marker="o", label="Train Top-1")
plt.plot(rounds, [h["val_top1"] for h in history], marker="s", label="Val Top-1")
plt.xlabel("Round")
plt.ylabel("Top-1")
plt.title("Top-1 por round")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "top1_curve.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(rounds, [h["train_top5"] for h in history], marker="o", label="Train Top-5")
plt.plot(rounds, [h["val_top5"] for h in history], marker="s", label="Val Top-5")
plt.xlabel("Round")
plt.ylabel("Top-5")
plt.title("Top-5 por round")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "top5_curve.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(rounds, [h["epoch_time"] for h in history], marker="o")
plt.xlabel("Round")
plt.ylabel("Segundos")
plt.title("Tiempo por round")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "time_per_round.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(rounds, [h["total_time"] for h in history], marker="o")
plt.xlabel("Round")
plt.ylabel("Segundos")
plt.title("Tiempo acumulado")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "time_cumulative.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(rounds, [h["train_top1"] - h["val_top1"] for h in history], marker="o")
plt.xlabel("Round")
plt.ylabel("Gap Top-1")
plt.title("Gap de generalización Top-1")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "generalization_gap_top1.png")
plt.close()

generate_analysis_notebook(
    results_dir=str(OUT_DIR),
    summary=summary,
    history_csv_path=str(csv_path),
    summary_json_path=str(json_path),
)

print(f"Demo generada en: {OUT_DIR}")
