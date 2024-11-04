from typing import Dict, List, Any

import numpy as np
import torch
from torch.utils.data import Dataset


class StandardDataset(Dataset):
    """Dataset for standard models (Neural Net, Monte Carlo)"""

    def __init__(
            self,
            features: Dict[str, np.ndarray],
            targets: np.ndarray
    ):
        """
        Initialize StandardDataset

        Args:
            features: Dictionary containing features
                - 'team_stats': Team statistics
                - 'rankings': Team rankings
                - 'form': Form metrics
                - 'advanced': Advanced statistics
            targets: Target values (win/loss, over/under)
        """
        # Convert all features to tensors
        self.features = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in features.items()
        }

        self.targets = torch.tensor(targets, dtype=torch.float32)

        # Prepare combined feature tensor
        self.combined_features = self._combine_features()

        # Validate shapes
        self._validate_shapes()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset

        Returns dictionary containing:
            - features: Combined feature tensor
            - feature_dict: Dictionary of individual feature groups
            - target: Target values
        """
        return {
            'features': self.combined_features[idx],
            'feature_dict': {
                k: v[idx] for k, v in self.features.items()
            },
            'target': self.targets[idx]
        }

    def _combine_features(self) -> torch.Tensor:
        """Combine all features into a single tensor"""
        feature_list = [v for v in self.features.values()]
        return torch.cat(feature_list, dim=-1)

    def _validate_shapes(self):
        """Validate shapes of input tensors"""
        # Get primary length
        primary_len = len(self.targets)

        # Check feature shapes
        for k, v in self.features.items():
            assert len(v) == primary_len, f"Feature {k} length mismatch"

        # Check combined features
        assert len(self.combined_features) == primary_len, "Combined features length mismatch"

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for DataLoader

        Args:
            batch: List of dictionaries containing feature data

        Returns:
            Dictionary containing batched tensors
        """
        return {
            'features': torch.stack([b['features'] for b in batch]),
            'feature_dict': {
                k: torch.stack([b['feature_dict'][k] for b in batch])
                for k in batch[0]['feature_dict'].keys()
            },
            'target': torch.stack([b['target'] for b in batch])
        }
