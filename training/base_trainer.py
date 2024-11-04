from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, List, Optional

from torch.utils.data import DataLoader

from .callbacks.callback_manager import CallbackManager
from .validation.metrics_tracker import MetricsTracker
from .validation.early_stopping import EarlyStopping

class BaseTrainer(ABC):
    def __init__(
            self,
            model: torch.nn.Module,
            config: Any,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: Optional[DataLoader] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Setup device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(
            save_dir=config.SAVE_DIR,
            model_name=self.model.__class__.__name__
        )

        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_DELTA,
            monitor=config.EARLY_STOPPING_MONITOR
        )

        # Initialize callback manager
        self.callback_manager = CallbackManager(self.config)
        self._init_callbacks()

        # Initialize optimizer and criterion
        self.optimizer = self._init_optimizer()
        self.criterion = self._init_criterion()

    @abstractmethod
    def _init_optimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def _init_criterion(self) -> Any:
        pass

    def _init_callbacks(self):
        """Initialize training callbacks"""
        from .callbacks.model_checkpoint import ModelCheckpoint
        from .callbacks.learning_rate import LearningRateScheduler
        from .callbacks.tensorboard_logger import TensorboardLogger

        # Add model checkpoint callback
        self.callback_manager.add_callback(
            ModelCheckpoint(
                model_manager=self.config.MODEL_MANAGER,
                monitor=self.config.CHECKPOINT_MONITOR,
                mode=self.config.CHECKPOINT_MODE
            )
        )

        # Add learning rate scheduler
        self.callback_manager.add_callback(
            LearningRateScheduler(
                optimizer=self.optimizer,
                scheduler_type=self.config.LR_SCHEDULER,
                **self.config.LR_SCHEDULER_PARAMS
            )
        )

        # Add tensorboard logger
        self.callback_manager.add_callback(
            TensorboardLogger(
                log_dir=f"{self.config.LOG_DIR}/{self.model.__class__.__name__}"
            )
        )

    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        # Notify training start
        self.callback_manager.on_train_begin(self._get_logs())

        for epoch in range(self.config.EPOCHS):
            # Notify epoch start
            self.callback_manager.on_epoch_begin(epoch, self._get_logs())

            # Training phase
            train_metrics = self._train_epoch()

            # Validation phase
            val_metrics = self._validate_epoch()

            # Update metrics
            self.metrics_tracker.update_metrics(
                train_metrics,
                val_metrics,
                epoch
            )

            # Update logs and notify callbacks
            logs = self._get_logs()
            logs.update({
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'epoch': epoch
            })
            self.callback_manager.on_epoch_end(epoch, logs)

            # Early stopping check
            if self.early_stopping(val_metrics[self.config.EARLY_STOPPING_MONITOR]):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Final evaluation
        final_metrics = self._final_evaluation()

        # Notify training end
        self.callback_manager.on_train_end(self._get_logs(final_metrics))

        return final_metrics

    def _get_logs(self, additional_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get current training logs"""
        logs = {
            'model': self.model,
            'optimizer': self.optimizer,
            'model_type': self.model.__class__.__name__,
            'current_metrics': self.metrics_tracker.get_current_metrics(),
            'best_metrics': self.metrics_tracker.get_best_metrics()
        }

        if additional_metrics:
            logs.update(additional_metrics)

        return logs

    def _final_evaluation(self) -> Dict[str, Any]:
        """Perform final model evaluation"""
        metrics = {
            'train_metrics': self.metrics_tracker.get_best_metrics(),
            'training_history': {
                'train': self.metrics_tracker.train_metrics,
                'val': self.metrics_tracker.val_metrics
            }
        }

        # Test set evaluation if available
        if self.test_loader is not None:
            test_metrics = self._evaluate(self.test_loader)
            metrics['test_metrics'] = test_metrics

        return metrics

    @abstractmethod
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        pass

    @abstractmethod
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        pass

    def _evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on given data"""
        self.model.eval()
        metrics = {}

        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch['target'])

                # Update metrics
                batch_metrics = self._calculate_metrics(outputs, batch['target'])
                for name, value in batch_metrics.items():
                    metrics[name] = metrics.get(name, 0) + value

        # Average metrics
        return {k: v / len(data_loader) for k, v in metrics.items()}

    @abstractmethod
    def _calculate_metrics(
            self,
            outputs: torch.Tensor,
            targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate metrics for a batch"""
        pass
