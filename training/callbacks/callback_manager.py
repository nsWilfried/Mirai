from typing import Dict, Any, List

from training.callbacks.base_callback import Callback


class CallbackManager:
    def __init__(self, config: Any):
        self.callbacks: List[Callback] = []
        self.config = config

    def add_callback(self, callback: Callback):
        self.callbacks.append(callback)

    def on_train_begin(self, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
