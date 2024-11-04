from typing import Optional, Any, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from training.base_trainer import BaseTrainer


class LSTMTrainer(BaseTrainer):
    def __init__(
            self,
            model: torch.nn.Module,
            config: Any,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: Optional[DataLoader] = None
    ):
        super().__init__(model, config, train_loader, val_loader, test_loader)

    def _init_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

    def _init_criterion(self) -> Any:
        return torch.nn.BCEWithLogitsLoss()

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []

        for batch in self.train_loader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch['target'])

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.GRAD_CLIP
            )

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions.extend(outputs.detach().cpu().numpy())
            targets.extend(batch['target'].cpu().numpy())

        # Calculate metrics
        metrics = self._calculate_metrics(
            np.array(predictions),
            np.array(targets)
        )
        metrics['loss'] = total_loss / len(self.train_loader)

        return metrics

    def _validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch['target'])

                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch['target'].cpu().numpy())

        metrics = self._calculate_metrics(
            np.array(predictions),
            np.array(targets)
        )
        metrics['loss'] = total_loss / len(self.val_loader)

        return metrics

    def _calculate_metrics(
            self,
            outputs: np.ndarray,
            targets: np.ndarray
    ) -> Dict[str, float]:
        predictions = (outputs >= 0.5).astype(int)
        return {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='weighted'),
            'recall': recall_score(targets, predictions, average='weighted'),
            'f1': f1_score(targets, predictions, average='weighted'),
            'auc': roc_auc_score(targets, outputs)
        }
