from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def plot_yolo_metrics(base_dir, output_dir="plots", experiments_to_compare=None):
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    experiment_data = {}
    
    for experiment in base_dir.glob("*"):
        if experiment.is_dir() and "weights" not in experiment.name:
            results_file = experiment / "results.csv"
            
            if results_file.exists():
                print(f"Обрабатываю эксперимент: {experiment.name}")
                df = pd.read_csv(results_file)
                
                experiment_data[experiment.name] = {
                    "epochs": df["epoch"].values if "epoch" in df.columns else np.arange(len(df)),
                    "train_loss": df.get("train/box_loss", np.zeros(len(df))),
                    "val_loss": df.get("val/box_loss", np.zeros(len(df))),
                    "mAP50": df.get("metrics/mAP50(B)", np.zeros(len(df))),
                    "mAP50_95": df.get("metrics/mAP50-95(B)", np.zeros(len(df))),
                    "precision": df.get("metrics/precision(B)", np.zeros(len(df))),
                    "recall": df.get("metrics/recall(B)", np.zeros(len(df)))
                }
            else:
                print(f"Предупреждение: Файл {results_file} не найден")
    
    if not experiment_data:
        print("Ошибка: Не найдено данных для построения графиков")
        return
    

    if experiments_to_compare:
        experiment_data = {k: v for k, v in experiment_data.items() if k in experiments_to_compare}
    
    _plot_loss_curves(experiment_data, output_dir)
    _plot_map_metrics(experiment_data, output_dir)
    _plot_prf_metrics(experiment_data, output_dir)
    _plot_comparison_summary(experiment_data, output_dir)
    
    print(f"Все графики сохранены в {output_dir}")


def _plot_loss_curves(experiment_data, output_dir):
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_data)))
    
    for i, (name, data) in enumerate(experiment_data.items()):
        color = colors[i]
        plt.plot(data["epochs"], data["train_loss"], 
                label=f"Train Loss ({name})", color=color, linestyle='-')
        plt.plot(data["epochs"], data["val_loss"], 
                label=f"Val Loss ({name})", color=color, linestyle='--')
    
    plt.title("Training and Validation Loss", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / "loss_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ График Loss сохранён: loss_curves.png")


def _plot_map_metrics(experiment_data, output_dir):
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_data)))
    
    for i, (name, data) in enumerate(experiment_data.items()):
        color = colors[i]
        plt.plot(data["epochs"], data["mAP50"], 
                label=f"mAP@50 ({name})", color=color, linestyle='-', linewidth=2)
        plt.plot(data["epochs"], data["mAP50_95"], 
                label=f"mAP@50-95 ({name})", color=color, linestyle=':', linewidth=2)
    
    plt.title("Mean Average Precision (mAP)", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    plt.savefig(output_dir / "map_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ График mAP сохранён: map_metrics.png")


def _plot_prf_metrics(experiment_data, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_data)))
    
    for i, (name, data) in enumerate(experiment_data.items()):
        color = colors[i]
        
        precision = data["precision"]
        recall = data["recall"]
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        axes[0].plot(data["epochs"], precision, label=name, color=color, linewidth=2)
        axes[0].set_title("Precision", fontweight='bold')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Precision")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        axes[1].plot(data["epochs"], recall, label=name, color=color, linewidth=2)
        axes[1].set_title("Recall", fontweight='bold')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Recall")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        axes[2].plot(data["epochs"], f1, label=name, color=color, linewidth=2)
        axes[2].set_title("F1-Score", fontweight='bold')
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("F1-Score")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_f1.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ График P/R/F1 сохранён: precision_recall_f1.png")


def _plot_comparison_summary(experiment_data, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_data)))
    
    for i, (name, data) in enumerate(experiment_data.items()):
        color = colors[i]
        
        axes[0, 0].plot(data["epochs"], data["mAP50"], 
                       label=name, color=color, linewidth=2)
        axes[0, 0].set_title("mAP@50 Comparison", fontweight='bold')
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("mAP@50")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(data["epochs"], data["precision"], 
                       label=name, color=color, linewidth=2)
        axes[0, 1].set_title("Precision Comparison", fontweight='bold')
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Precision")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        

        axes[1, 0].plot(data["epochs"], data["recall"], 
                       label=name, color=color, linewidth=2)
        axes[1, 0].set_title("Recall Comparison", fontweight='bold')
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Recall")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(data["epochs"], data["val_loss"], 
                       label=name, color=color, linewidth=2)
        axes[1, 1].set_title("Validation Loss Comparison", fontweight='bold')
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Val Loss")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "experiments_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Сравнительный график сохранён: experiments_comparison.png")


def generate_metrics_report(experiment_data, output_dir):
    output_dir = Path(output_dir)
    
    with open(output_dir / "metrics_report.txt", "w", encoding="utf-8") as f:
        f.write("ОТЧЁТ ПО МЕТРИКАМ YOLOV11\n")
        f.write("=" * 50 + "\n\n")
        
        for name, data in experiment_data.items():
            f.write(f"Эксперимент: {name}\n")
            f.write("-" * 30 + "\n")
            
            final_epoch = data["epochs"][-1]
            final_map50 = data["mAP50"][-1]
            final_map50_95 = data["mAP50_95"][-1]
            final_precision = data["precision"][-1]
            final_recall = data["recall"][-1]
            final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall + 1e-10)
            
            f.write(f"Финальная эпоха: {final_epoch}\n")
            f.write(f"mAP@50: {final_map50:.4f}\n")
            f.write(f"mAP@50-95: {final_map50_95:.4f}\n")
            f.write(f"Precision: {final_precision:.4f}\n")
            f.write(f"Recall: {final_recall:.4f}\n")
            f.write(f"F1-Score: {final_f1:.4f}\n\n")
    
    print("✓ Отчёт сохранён: metrics_report.txt")