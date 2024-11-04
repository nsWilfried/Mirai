from abc import ABC, abstractmethod
from typing import Dict, Any

class Callback(ABC):
    @abstractmethod
    def on_train_begin(self, logs: Dict[str, Any] = None):
        pass

    @abstractmethod
    def on_train_end(self, logs: Dict[str, Any] = None):
        pass

    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None):
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        pass
