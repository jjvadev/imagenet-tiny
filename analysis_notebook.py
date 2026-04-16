import json
import os


def generate_analysis_notebook(
    results_dir: str,
    summary: dict,
    history_csv_path: str,
    summary_json_path: str,
):
    nb_path = os.path.join(results_dir, "analysis.ipynb")

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Analisis de entrenamiento Tiny ImageNet\n",
                    "\n",
                    "Notebook generado automaticamente.\n",
                    "\n",
                    "Metricas minimas:\n",
                    "- epoch\n",
                    "- cost\n",
                    "- train\n",
                    "- test\n",
                    "- epoch_time\n",
                    "- total_time\n",
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
                    f'csv_file = Path(r"{history_csv_path}")\n',
                    f'json_file = Path(r"{summary_json_path}")\n',
                    "df = pd.read_csv(csv_file)\n",
                    "with open(json_file, 'r', encoding='utf-8') as f:\n",
                    "    summary = json.load(f)\n",
                    "df\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Resumen\n",
                    "\n",
                    f"- Arquitectura: {summary['arch']}\n",
                    f"- Optimizer: {summary['optimizer']}\n",
                    f"- Epochs: {summary['epochs_completed']}\n",
                    f"- Workers: {summary['num_workers']}\n",
                    f"- Local epochs: {summary['local_epochs']}\n",
                    f"- LR: {summary['lr']}\n",
                    f"- Batch size train: {summary['train_batch_size']}\n",
                    f"- Batch size test: {summary['val_batch_size']}\n",
                    f"- Device: {summary['device']}\n",
                    f"- Mejor test: {summary['best_test']:.4f}\n",
                    f"- Mejor epoch: {summary['best_epoch']}\n",
                    f"- Tiempo total: {summary['total_time_sec']:.2f} s\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "best_idx = df['test'].idxmax()\n",
                    "best_row = df.loc[best_idx]\n",
                    "print(f\"Mejor epoch: {int(best_row['epoch'])}\")\n",
                    "print(f\"Cost: {best_row['cost']:.6f}\")\n",
                    "print(f\"Train: {best_row['train']:.6f}\")\n",
                    "print(f\"Test: {best_row['test']:.6f}\")\n",
                    "print(f\"Tiempo total final: {df.iloc[-1]['total_time']:.2f} s\")\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(8,5))\n",
                    "plt.plot(df['epoch'], df['cost'], marker='o')\n",
                    "plt.title('Cost por epoch')\n",
                    "plt.xlabel('Epoch')\n",
                    "plt.ylabel('Cost')\n",
                    "plt.grid(True)\n",
                    "plt.show()\n",
                    "\n",
                    "plt.figure(figsize=(8,5))\n",
                    "plt.plot(df['epoch'], df['train'], marker='o', label='Train')\n",
                    "plt.plot(df['epoch'], df['test'], marker='s', label='Test')\n",
                    "plt.title('Accuracy por epoch')\n",
                    "plt.xlabel('Epoch')\n",
                    "plt.ylabel('Accuracy')\n",
                    "plt.grid(True)\n",
                    "plt.legend()\n",
                    "plt.show()\n",
                    "\n",
                    "plt.figure(figsize=(8,5))\n",
                    "plt.plot(df['epoch'], df['epoch_time'], marker='o', label='Tiempo por epoch')\n",
                    "plt.plot(df['epoch'], df['total_time'], marker='s', label='Tiempo acumulado')\n",
                    "plt.title('Tiempo de entrenamiento')\n",
                    "plt.xlabel('Epoch')\n",
                    "plt.ylabel('Segundos')\n",
                    "plt.grid(True)\n",
                    "plt.legend()\n",
                    "plt.show()\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"Notebook generado: {nb_path}")