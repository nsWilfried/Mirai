# models/lstm/lstm_model.py
from typing import Dict, Tuple

import torch
from torch import nn

from config.config import Config
from models.base.base_model import BaseModel


class TeamLSTM(BaseModel):
    def __init__(self, config: Config):
        super().__init__(config)
        params = config.LSTM_PARAMS

        # Process sequence data
        self.lstm = nn.LSTM(
            input_size=len(config.KEY_STATS),
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            bidirectional=params["bidirectional"]
        )

        # Process basic stats
        self.basic_stats_encoder = nn.Sequential(
            nn.Linear(len(config.BASIC_STATS), params["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(params["dropout"])
        )

        # Process rankings
        self.rankings_encoder = nn.Sequential(
            nn.Linear(len(config.RANK_FEATURES), params["hidden_size"] // 2),
            nn.ReLU(),
            nn.Dropout(params["dropout"])
        )

        # Process moving averages
        moving_avgs_size = len(config.KEY_STATS) * len(config.ROLLING_WINDOWS)
        self.moving_avgs_encoder = nn.Sequential(
            nn.Linear(moving_avgs_size, params["hidden_size"] // 2),
            nn.ReLU(),
            nn.Dropout(params["dropout"])
        )

        # Calculate combined size
        lstm_out_size = params["hidden_size"] * (2 if params["bidirectional"] else 1)
        combined_size = (
                lstm_out_size +        # LSTM output
                params["hidden_size"] + # Basic stats
                params["hidden_size"] // 2 + # Rankings
                params["hidden_size"] // 2 + # Moving averages
                1                      # Rest feature
        )

        # Final layers
        self.fc = nn.Sequential(
            nn.Linear(combined_size, params["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["hidden_size"], len(config.TARGET_COLUMNS))
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process sequence through LSTM
        lstm_out, _ = self.lstm(x['sequence'])
        lstm_features = lstm_out[:, -1, :]

        # Process basic stats
        basic_features = self.basic_stats_encoder(x['basic_stats'])

        # Process rankings
        ranking_features = self.rankings_encoder(x['rankings'])

        # Process moving averages
        moving_avg_features = self.moving_avgs_encoder(x['moving_averages'])

        # Combine all features
        combined = torch.cat([
            lstm_features,
            basic_features,
            ranking_features,
            moving_avg_features,
            x['rest']
        ], dim=1)

        # Final prediction
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
