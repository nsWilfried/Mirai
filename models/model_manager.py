# training/model_management/model_manager.py
from pathlib import Path
import torch
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import os

class ModelManager:
    def __init__(self, config: Any):
        self.config = config
        self.base_path = Path(config.MODEL_SAVE_DIR)
        self.models_dir = self.base_path / "models"
        self.metadata_dir = self.base_path / "metadata"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Track best models
        self.best_models = {}
        self._load_best_models_info()

    def save_model(
            self,
            model: torch.nn.Module,
            model_type: str,
            metrics: Dict[str, float],
            epoch: int,
            is_best: bool = False
    ) -> str:
        """Save model with metadata"""
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create model ID
        model_id = f"{model_type}_{timestamp}"

        # Save model state
        model_path = self.models_dir / f"{model_id}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'config': self.config.__dict__,
            'epoch': epoch
        }, model_path)

        # Save metadata
        metadata = {
            'model_id': model_id,
            'model_type': model_type,
            'timestamp': timestamp,
            'metrics': metrics,
            'epoch': epoch,
            'is_best': is_best
        }

        metadata_path = self.metadata_dir / f"{model_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        # Update best model if necessary
        if is_best:
            self._update_best_model(model_id, model_type, metrics)

        return model_id

    def load_best_model(
            self,
            model_type: str,
            device: Optional[torch.device] = None
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Load the best model for a specific type"""
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        if model_type not in self.best_models:
            raise ValueError(f"No best model found for type {model_type}")

        model_id = self.best_models[model_type]['model_id']
        return self.load_model(model_id, device)

    def load_model(
            self,
            model_id: str,
            device: torch.device
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Load a specific model by ID"""
        model_path = self.models_dir / f"{model_id}.pth"
        metadata_path = self.metadata_dir / f"{model_id}.json"

        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Model {model_id} not found")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Initialize model
        model_class = globals()[checkpoint['model_class']]
        model = model_class(Config(**checkpoint['config']))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        return model, metadata

    def _update_best_model(
            self,
            model_id: str,
            model_type: str,
            metrics: Dict[str, float]
    ):
        """Update best model tracking"""
        # Create best model symlink
        best_model_path = self.models_dir / f"best_{model_type}.pth"
        best_metadata_path = self.metadata_dir / f"best_{model_type}.json"

        # Remove old symlinks if they exist
        if best_model_path.exists():
            best_model_path.unlink()
        if best_metadata_path.exists():
            best_metadata_path.unlink()

        # Create new symlinks
        os.symlink(
            self.models_dir / f"{model_id}.pth",
            best_model_path
        )
        os.symlink(
            self.metadata_dir / f"{model_id}.json",
            best_metadata_path
        )

        # Update best models tracking
        self.best_models[model_type] = {
            'model_id': model_id,
            'metrics': metrics
        }

        # Save best models info
        self._save_best_models_info()

    def _save_best_models_info(self):
        """Save best models information"""
        info_path = self.base_path / "best_models.json"
        with open(info_path, 'w') as f:
            json.dump(self.best_models, f, indent=4)

    def _load_best_models_info(self):
        """Load best models information"""
        info_path = self.base_path / "best_models.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.best_models = json.load(f)
