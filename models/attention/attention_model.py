from typing import Dict, Tuple
import torch
from torch import nn
from config.config import Config
from models.base.base_model import BaseModel


# models/attention/attention_model.py
class MultiHeadAttentionModel(BaseModel):
    def __init__(self, config: Config):
        super().__init__(config)
        params = config.ATTENTION_PARAMS

        # Multi-head attention for sequences
        self.sequence_attention = nn.MultiheadAttention(
            embed_dim=len(config.KEY_STATS),
            num_heads=params["num_heads"],
            dropout=params["dropout"]
        )

        # Transformer encoder for sequences
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=len(config.KEY_STATS),
            nhead=params["num_heads"],
            dim_feedforward=params["hidden_size"],
            dropout=params["dropout"]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=params["num_layers"]
        )

        # Process basic stats
        self.basic_stats_encoder = nn.Sequential(
            nn.Linear(len(config.BASIC_STATS), params["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(params["dropout"])
        )

        # Process rankings with self-attention
        self.rankings_attention = nn.MultiheadAttention(
            embed_dim=len(config.RANK_FEATURES),
            num_heads=params["num_heads"] // 2,
            dropout=params["dropout"]
        )

        # Moving averages processing
        moving_avgs_size = len(config.KEY_STATS) * len(config.ROLLING_WINDOWS)
        self.moving_avgs_encoder = nn.Sequential(
            nn.Linear(moving_avgs_size, params["hidden_size"] // 2),
            nn.ReLU(),
            nn.Dropout(params["dropout"])
        )

        # Combined processing
        combined_size = (
                len(config.KEY_STATS) +   # Transformed sequence
                params["hidden_size"] +    # Basic stats
                len(config.RANK_FEATURES) + # Rankings
                params["hidden_size"] // 2 + # Moving averages
                1                          # Rest feature
        )

        self.fc = nn.Sequential(
            nn.Linear(combined_size, params["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["hidden_size"], len(config.TARGET_COLUMNS))
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process sequence with attention
        sequence_attended, _ = self.sequence_attention(
            x['sequence'],
            x['sequence'],
            x['sequence']
        )
        sequence_encoded = self.transformer_encoder(sequence_attended)
        sequence_features = sequence_encoded.mean(dim=1)

        # Process basic stats
        basic_features = self.basic_stats_encoder(x['basic_stats'])

        # Process rankings with attention
        rankings_attended, _ = self.rankings_attention(
            x['rankings'].unsqueeze(0),
            x['rankings'].unsqueeze(0),
            x['rankings'].unsqueeze(0)
        )
        ranking_features = rankings_attended.squeeze(0)

        # Process moving averages
        moving_avg_features = self.moving_avgs_encoder(x['moving_averages'])

        # Combine features
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
