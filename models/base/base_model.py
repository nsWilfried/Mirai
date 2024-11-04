from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.nn as nn

from config.config import Config


class BaseModel(ABC, nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(
            self,
            x: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        """Return predictions and confidence scores"""
        pass

    def calculate_confidence(self, output: torch.Tensor) -> float:
        """Calculate prediction confidence"""
        # Base confidence on prediction certainty
        probabilities = torch.sigmoid(output)
        confidence = torch.abs(probabilities - 0.5) * 2
        return confidence.mean().item()
