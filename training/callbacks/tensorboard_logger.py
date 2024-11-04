from abc import ABC
from typing import Dict, Any

from torch.utils.tensorboard import SummaryWriter

from training.callbacks.base_callback import Callback


class TensorboardLogger(Callback, ABC):
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def on_train_begin(self, logs: Dict[str, Any] = None):
        if logs and 'model' in logs:
            self.writer.add_graph(
                logs['model'],
                logs.get('example_input', None)
            )

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        if logs is None:
            return

        # Log metrics
        for metric_name, value in logs.get('metrics', {}).items():
            self.writer.add_scalar(f'metrics/{metric_name}', value, epoch)

        # Log learning rates
        if 'optimizer' in logs:
            for i, param_group in enumerate(logs['optimizer'].param_groups):
                self.writer.add_scalar(
                    f'learning_rate/group_{i}',
                    param_group['lr'],
                    epoch
                )

        # Log model weights histograms
        if 'model' in logs:
            for name, param in logs['model'].named_parameters():
                self.writer.add_histogram(
                    f'parameters/{name}',
                    param.data.cpu().numpy(),
                    epoch
                )
