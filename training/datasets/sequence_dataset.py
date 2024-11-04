# data/datasets.py
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Any, List, Tuple

class SequenceDataset(Dataset):
    """Dataset for sequence-based models (LSTM, Attention)"""

    def __init__(
            self,
            sequences: Dict[str, torch.Tensor],
            features: Dict[str, np.ndarray],
            targets: np.ndarray
    ):
        """
        Initialize SequenceDataset

        Args:
            sequences: Dictionary containing sequence data
                - 'sequences': Game sequence features
                - 'features': Static features
                - 'targets': Target values
            features: Dictionary containing additional features
                - 'team_stats': Team statistics
                - 'rankings': Team rankings
                - 'form': Form metrics
                - 'advanced': Advanced statistics
            targets: Target values (win/loss, over/under)
        """
        self.sequences = {
            k: torch.tensor(v, dtype=torch.float32)
            if not torch.is_tensor(v) else v
            for k, v in sequences.items()
        }

        self.features = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in features.items()
        }

        self.targets = torch.tensor(targets, dtype=torch.float32)

        # Validate shapes
        self._validate_shapes()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset

        Returns dictionary containing:
            - sequence: Sequence of game features
            - features: Additional game features
            - target: Target values
        """
        return {
            'sequence': {
                k: v[idx] for k, v in self.sequences.items()
            },
            'features': {
                k: v[idx] for k, v in self.features.items()
            },
            'target': self.targets[idx]
        }

    def _validate_shapes(self):
        """Validate shapes of input tensors"""
        # Get primary sequence length
        seq_len = len(self.targets)

        # Check sequence shapes
        for k, v in self.sequences.items():
            assert len(v) == seq_len, f"Sequence {k} length mismatch"

        # Check feature shapes
        for k, v in self.features.items():
            assert len(v) == seq_len, f"Feature {k} length mismatch"

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for DataLoader

        Args:
            batch: List of dictionaries containing sequence data

        Returns:
            Dictionary containing batched tensors
        """
        return {
            'sequence': {
                k: torch.stack([b['sequence'][k] for b in batch])
                for k in batch[0]['sequence'].keys()
            },
            'features': {
                k: torch.stack([b['features'][k] for b in batch])
                for k in batch[0]['features'].keys()
            },
            'target': torch.stack([b['target'] for b in batch])
        }
