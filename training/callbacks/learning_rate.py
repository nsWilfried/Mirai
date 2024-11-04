from abc import ABC
from typing import Dict, Any

import torch

from training.callbacks.base_callback import Callback

class LearningRateScheduler(Callback, ABC):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            scheduler_type: str,
            **scheduler_params
    ):
        self.optimizer = optimizer
        self.scheduler = self._init_scheduler(
            scheduler_type,
            scheduler_params
        )

    def _init_scheduler(
            self,
            scheduler_type: str,
            params: Dict[str, Any]
    ) -> Any:
        if scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **params
            )
        elif scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                **params
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                **params
            )
