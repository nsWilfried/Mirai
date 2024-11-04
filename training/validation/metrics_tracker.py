# training/validation/metrics_tracker.py
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import json
from pathlib import Path

class MetricsTracker:
    def __init__(self, save_dir: str, model_name: str):
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics storage
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.best_metrics = {}
        self.current_epoch = 0

        # Initialize tensorboard writer
        self.writer = SummaryWriter(
            str(self.save_dir / 'tensorboard' / model_name)
        )

    def update_metrics(
            self,
            train_metrics: Dict[str, float],
            val_metrics: Dict[str, float],
            epoch: int
    ):
        """Update all metrics"""
        self.current_epoch = epoch

        # Update training metrics
        for metric, value in train_metrics.items():
            self.train_metrics[metric].append(value)
            self.writer.add_scalar(f'train/{metric}', value, epoch)

        # Update validation metrics
        for metric, value in val_metrics.items():
            self.val_metrics[metric].append(value)
            self.writer.add_scalar(f'val/{metric}', value, epoch)

            # Update best metrics
            if metric not in self.best_metrics or value > self.best_metrics[metric]:
                self.best_metrics[metric] = value
                self.writer.add_scalar(f'best/{metric}', value, epoch)

        # Save metrics to file
        self._save_metrics()

    def _save_metrics(self):
        """Save metrics to JSON file"""
        metrics_data = {
            'train': dict(self.train_metrics),
            'val': dict(self.val_metrics),
            'best': self.best_metrics,
            'current_epoch': self.current_epoch
        }

        metrics_file = self.save_dir / f'{self.model_name}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=4)

    def plot_metrics(self, metrics: List[str] = None):
        """Plot specified metrics"""
        if metrics is None:
            metrics = list(self.train_metrics.keys())

        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_metrics[metric], label=f'Train {metric}')
            plt.plot(self.val_metrics[metric], label=f'Val {metric}')
            plt.title(f'{metric} Over Time - {self.model_name}')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            plt.savefig(self.save_dir / f'{self.model_name}_{metric}_plot.png')
            plt.close()

    def get_current_metrics(self) -> Dict[str, float]:
        """Get most recent metrics"""
        return {
            'train': {k: v[-1] for k, v in self.train_metrics.items()},
            'val': {k: v[-1] for k, v in self.val_metrics.items()}
        }

    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics achieved"""
        return self.best_metrics.copy()
