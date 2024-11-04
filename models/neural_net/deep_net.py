# models/neural_net/deep_net.py
from typing import Dict, Tuple

import torch
from torch import nn

from config.config import Config
from models.base.base_model import BaseModel


class DeepNeuralNet(BaseModel):
    def __init__(self, config: Config):
        super().__init__(config)
        params = config.NEURAL_NET_PARAMS

        # Process sequence data
        sequence_input_size = len(config.KEY_STATS) * config.SEQUENCE_LENGTH
        self.sequence_encoder = nn.Sequential(
            nn.Linear(sequence_input_size, params["hidden_sizes"][0]),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["hidden_sizes"][0], params["hidden_sizes"][1]),
            nn.ReLU(),
            nn.Dropout(params["dropout"])
        )

        # Process basic stats
        self.basic_stats_encoder = nn.Sequential(
            nn.Linear(len(config.BASIC_STATS), params["hidden_sizes"][1]),
            nn.ReLU(),
            nn.Dropout(params["dropout"])
        )

        # Process rankings
        self.rankings_encoder = nn.Sequential(
            nn.Linear(len(config.RANK_FEATURES), params["hidden_sizes"][1] // 2),
            nn.ReLU(),
            nn.Dropout(params["dropout"])
        )

        # Process moving averages
        moving_avgs_size = len(config.KEY_STATS) * len(config.ROLLING_WINDOWS)
        self.moving_avgs_encoder = nn.Sequential(
            nn.Linear(moving_avgs_size, params["hidden_sizes"][1] // 2),
            nn.ReLU(),
            nn.Dropout(params["dropout"])
        )

        # Combined processing
        combined_size = (
                params["hidden_sizes"][1] +     # Sequence features
                params["hidden_sizes"][1] +     # Basic stats
                params["hidden_sizes"][1] // 2 + # Rankings
                params["hidden_sizes"][1] // 2 + # Moving averages
                1                               # Rest feature
        )

        # Final layers
        self.fc = nn.Sequential(
            nn.Linear(combined_size, params["hidden_sizes"][1]),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["hidden_sizes"][1], params["hidden_sizes"][2]),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["hidden_sizes"][2], len(config.TARGET_COLUMNS))
        )

        self.batch_norm_layers = nn.ModuleList([
            nn.BatchNorm1d(size) for size in params["hidden_sizes"]
        ])

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Flatten and process sequence
        batch_size = x['sequence'].size(0)
        sequence_flat = x['sequence'].view(batch_size, -1)
        sequence_features = self.sequence_encoder(sequence_flat)
        sequence_features = self.batch_norm_layers[0](sequence_features)

        # Process basic stats
        basic_features = self.basic_stats_encoder(x['basic_stats'])
        basic_features = self.batch_norm_layers[1](basic_features)

        # Process rankings
        ranking_features = self.rankings_encoder(x['rankings'])

        # Process moving averages
        moving_avg_features = self.moving_avgs_encoder(x['moving_averages'])

        # Combine all features
        combined = torch.cat([
            sequence_features,
            basic_features,
            ranking_features,
            moving_avg_features,
            x['rest']
        ], dim=1)

        return self.fc(combined)

    def predict(
            self,
            x: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        self.eval()
        with torch.no_grad():
            output = self(x)
            confidence = self.calculate_confidence(output)
        return torch.sigmoid(output), confidence
