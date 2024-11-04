# models/monte_carlo/mc_simulator.py
from typing import Dict, Tuple

import numpy as np
import torch

from config.config import Config
from models.base.base_model import BaseModel
from models.neural_net.deep_net import DeepNeuralNet


class MonteCarloSimulator(BaseModel):
    def __init__(self, config: Config):
        super().__init__(config)
        # Use Neural Network as base model
        self.base_model = DeepNeuralNet(config)
        self.num_simulations = config.MONTE_CARLO_PARAMS["num_simulations"]
        self.noise_std = config.MONTE_CARLO_PARAMS.get("noise_std", 0.1)

    def add_noise(
            self,
            x: Dict[str, torch.Tensor],
            noise_scale: float = None
    ) -> Dict[str, torch.Tensor]:
        """Add random noise to features"""
        if noise_scale is None:
            noise_scale = self.noise_std

        noisy_x = {}
        for key, value in x.items():
            if key != 'rest':  # Don't add noise to rest days
                noise = torch.randn_like(value) * noise_scale
                noisy_x[key] = value + noise
            else:
                noisy_x[key] = value

        return noisy_x

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.base_model(x)

    def predict(
            self,
            x: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        self.eval()
        predictions = []

        with torch.no_grad():
            # Run multiple simulations
            for _ in range(self.num_simulations):
                # Add noise to features
                noisy_x = self.add_noise(x)

                # Get prediction
                output = self.forward(noisy_x)
                pred = torch.sigmoid(output)
                predictions.append(pred)

            # Stack predictions
            predictions = torch.stack(predictions)

            # Calculate mean prediction and uncertainty
            mean_pred = predictions.mean(dim=0)
            std_pred = predictions.std(dim=0)

            # Calculate confidence based on prediction stability
            confidence = 1.0 / (1.0 + torch.exp(-1.0 / std_pred.mean()))

            return mean_pred, confidence.item()

    def get_prediction_distribution(
            self,
            x: Dict[str, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """Get full distribution of predictions"""
        self.eval()
        predictions = []

        with torch.no_grad():
            for _ in range(self.num_simulations):
                noisy_x = self.add_noise(x)
                output = self.forward(noisy_x)
                pred = torch.sigmoid(output)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)

        return {
            'mean': predictions.mean(axis=0),
            'std': predictions.std(axis=0),
            'percentiles': {
                '5': np.percentile(predictions, 5, axis=0),
                '25': np.percentile(predictions, 25, axis=0),
                '75': np.percentile(predictions, 75, axis=0),
                '95': np.percentile(predictions, 95, axis=0)
            }
        }
