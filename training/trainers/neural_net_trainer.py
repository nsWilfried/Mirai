# training/trainers/neural_net_trainer.py
from typing import Any, Optional, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from training.base_trainer import BaseTrainer


class NeuralNetTrainer(BaseTrainer):
    def __init__(
            self,
            model: torch.nn.Module,
            config: Any,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: Optional[DataLoader] = None
    ):
        super().__init__(model, config, train_loader, val_loader, test_loader)

        # Initialize feature importance tracking
        self.feature_importance = {}

        # Initialize layer activation tracking
        self.activation_hooks = []
        self._register_activation_hooks()

    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer with layer-wise learning rates"""
        # Group parameters by layer for different learning rates
        params = []

        # Lower learning rate for early layers
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                params.append({
                    'params': param,
                    'lr': self.config.LEARNING_RATE * 0.1
                })
            else:
                params.append({
                    'params': param,
                    'lr': self.config.LEARNING_RATE
                })

        return torch.optim.AdamW(
            params,
            weight_decay=self.config.WEIGHT_DECAY,
            amsgrad=True
        )

    def _init_criterion(self) -> Any:
        """Initialize loss function with class weighting"""
        if hasattr(self.config, 'CLASS_WEIGHTS'):
            class_weights = torch.tensor(
                self.config.CLASS_WEIGHTS,
                device=self.device
            )
            return torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        return torch.nn.BCEWithLogitsLoss()

    def _register_activation_hooks(self):
        """Register hooks for layer activation tracking"""
        def hook_fn(name):
            def hook(module, input, output):
                if not hasattr(self, 'activations'):
                    self.activations = {}
                self.activations[name] = output.detach()
            return hook

        # Register hooks for each layer we want to track
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU)):
                self.activation_hooks.append(
                    module.register_forward_hook(hook_fn(name))
                )

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        feature_importances = []

        for batch in self.train_loader:
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

            # Optimizer step
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions.extend(outputs.detach().cpu().numpy())
            targets.extend(batch['target'].cpu().numpy())

            # Calculate feature importance for this batch
            if hasattr(self.model, 'feature_importances'):
                importance = self.model.feature_importances(batch)
                feature_importances.append(importance.cpu().numpy())

        # Calculate metrics
        metrics = self._calculate_metrics(
            np.array(predictions),
            np.array(targets)
        )
        metrics['loss'] = total_loss / len(self.train_loader)

        # Add feature importance metrics
        if feature_importances:
            self.feature_importance = self._calculate_feature_importance(
                np.mean(feature_importances, axis=0)
            )
            metrics['feature_importance'] = self.feature_importance

        # Add layer activation metrics
        activation_metrics = self._calculate_activation_metrics()
        metrics.update(activation_metrics)

        return metrics

    def _validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch['target'])

                # Track metrics
                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch['target'].cpu().numpy())

        # Calculate metrics
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
        """Calculate comprehensive metrics"""
        predictions = (outputs >= 0.5).astype(int)

        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='weighted'),
            'recall': recall_score(targets, predictions, average='weighted'),
            'f1': f1_score(targets, predictions, average='weighted'),
            'auc': roc_auc_score(targets, outputs)
        }

        # Add confusion matrix metrics
        cm = confusion_matrix(targets, predictions)
        metrics.update({
            'true_positives': cm[1, 1],
            'false_positives': cm[0, 1],
            'true_negatives': cm[0, 0],
            'false_negatives': cm[1, 0]
        })

        return metrics

    def _calculate_feature_importance(
            self,
            importance_scores: np.ndarray
    ) -> Dict[str, float]:
        """Calculate feature importance metrics"""
        if not hasattr(self.config, 'FEATURE_NAMES'):
            return {}

        feature_importance = {}
        for name, score in zip(self.config.FEATURE_NAMES, importance_scores):
            feature_importance[name] = float(score)

        return dict(
            sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )

    def _calculate_activation_metrics(self) -> Dict[str, float]:
        """Calculate layer activation metrics"""
        if not hasattr(self, 'activations'):
            return {}

        activation_metrics = {}
        for name, activation in self.activations.items():
            activation_metrics.update({
                f'{name}_mean': float(activation.mean()),
                f'{name}_std': float(activation.std()),
                f'{name}_dead_neurons': float(
                    (activation == 0).float().mean()
                )
            })

        return activation_metrics

    def cleanup(self):
        """Clean up resources"""
        # Remove activation hooks
        for hook in self.activation_hooks:
            hook.remove()

        # Clear activation storage
        if hasattr(self, 'activations'):
            del self.activations

# Example usage:
"""
# Initialize model and trainer
model = DeepNeuralNet(config)
trainer = NeuralNetTrainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)

# Train model
results = trainer.train()

# Plot feature importance
trainer.plot_feature_importance()

# Plot activation distributions
trainer.plot_activation_distributions()

# Clean up
trainer.cleanup()

# Access results
print("Training Results:")
print(f"Best Metrics: {results['train_metrics']}")
print(f"Test Metrics: {results['test_metrics']}")
print("\nFeature Importance:")
for feature, importance in results['train_metrics']['feature_importance'].items():
    print(f"{feature}: {importance:.4f}")
"""
