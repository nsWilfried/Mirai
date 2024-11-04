from abc import ABC
from typing import Any, Dict

from training.callbacks.base_callback import Callback


class ModelCheckpoint(Callback, ABC):
    def __init__(
            self,
            model_manager: Any,
            monitor: str,
            mode: str = 'min'
    ):
        self.model_manager = model_manager
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        if not logs:
            return

        current_value = logs['val_metrics'].get(self.monitor)
        if current_value is None:
            return

        if self.mode == 'min':
            is_best = current_value < self.best_value
        else:
            is_best = current_value > self.best_value

        if is_best:
            self.best_value = current_value
            self.model_manager.save_model(
                model=logs['model'],
                model_type=logs['model_type'],
                metrics=logs['val_metrics'],
                epoch=epoch,
                is_best=True
            )
