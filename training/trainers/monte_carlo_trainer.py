from typing import Any, Optional, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.base_trainer import BaseTrainer


class MonteCarloTrainer(BaseTrainer):
    def __init__(
            self,
            model: torch.nn.Module,
            config: Any,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: Optional[DataLoader] = None
    ):
        super().__init__(model, config, train_loader, val_loader, test_loader)

        # Initialize distribution tracking
        self.uncertainty_history = []

    def _init_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE
        )

    def _init_criterion(self) -> Any:
        return torch.nn.BCEWithLogitsLoss()

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        predictions = []
        uncertainties = []
        targets = []

        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()

            # Multiple forward passes with dropout
            mc_outputs = []
            for _ in range(self.config.MC_SAMPLES):
                outputs = self.model(batch)
                mc_outputs.append(outputs)

            # Calculate mean prediction and uncertainty
            mc_outputs = torch.stack(mc_outputs)
            mean_pred = mc_outputs.mean(dim=0)
            uncertainty = mc_outputs.std(dim=0)

            # Loss calculation
            loss = self.criterion(mean_pred, batch['target'])
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predictions.extend(mean_pred.detach().cpu().numpy())
            uncertainties.extend(uncertainty.detach().cpu().numpy())
            targets.extend(batch['target'].cpu().numpy())

        # Calculate metrics
        metrics = self._calculate_metrics(
            np.array(predictions),
            np.array(targets)
        )
        metrics['loss'] = total_loss / len(self.train_loader)

        # Add uncertainty metrics
        uncertainty_metrics = self._calculate_uncertainty_metrics(
            np.array(uncertainties),
            np.array(predictions),
            np.array(targets)
        )
        metrics.update(uncertainty_metrics)

        return metrics

    def _calculate_uncertainty_metrics(
            self,
            uncertainties: np.ndarray,
            predictions: np.ndarray,
            targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate uncertainty-related metrics"""
        correct_predictions = (predictions >= 0.5).astype(int) == targets

        return {
            'uncertainty_mean': np.mean(uncertainties),
            'uncertainty_std': np.std(uncertainties),
            'uncertainty_correct': np.mean(uncertainties[correct_predictions]),
            'uncertainty_incorrect': np.mean(uncertainties[~correct_predictions]),
            'calibration_score': self._calculate_calibration(
                predictions,
                uncertainties,
                targets
            )
        }

    def _calculate_calibration(
            self,
            predictions: np.ndarray,
            uncertainties: np.ndarray,
            targets: np.ndarray
    ) -> float:
        """Calculate model calibration score"""
        confidence_bins = np.linspace(0, 1, 11)
        calibration_scores = []

        for i in range(len(confidence_bins) - 1):
            mask = (
                    (1 - uncertainties >= confidence_bins[i]) &
                    (1 - uncertainties < confidence_bins[i + 1])
            )
            if np.any(mask):
                bin_accuracy = np.mean(
                    (predictions[mask] >= 0.5).astype(int) == targets[mask]
                )
                bin_confidence = np.mean(1 - uncertainties[mask])
                calibration_scores.append(abs(bin_accuracy - bin_confidence))

        return np.mean(calibration_scores) if calibration_scores else 1.0
