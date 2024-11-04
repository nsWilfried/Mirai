# preprocessing/sequence_builder.py
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


class SequenceBuilder:
    def __init__(self, config: Any):
        self.config = config
        self.sequence_length = config.SEQUENCE_LENGTH

    def build_sequences(
            self,
            data: pd.DataFrame,
            fold_indices: Optional[np.ndarray] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Build sequences for time series models

        Args:
            data: DataFrame containing game data
            fold_indices: Optional indices for cross-validation fold
        """
        if fold_indices is not None:
            data = data.iloc[fold_indices]

        sequences = []
        features = []
        targets = []

        for team_name, team_data in data.groupby('TEAM_NAME'):
            team_data = team_data.sort_values('Date')

            # Build sequences for this team
            team_sequences = self._build_team_sequences(team_data)

            sequences.extend(team_sequences['sequences'])
            features.extend(team_sequences['features'])
            targets.extend(team_sequences['targets'])

        return {
            'sequences': torch.tensor(sequences, dtype=torch.float32),
            'features': torch.tensor(features, dtype=torch.float32),
            'targets': torch.tensor(targets, dtype=torch.float32)
        }

    def _build_team_sequences(
            self,
            team_data: pd.DataFrame
    ) -> Dict[str, List]:
        """Build sequences for a single team"""
        sequences = []
        features = []
        targets = []

        for i in range(len(team_data) - self.sequence_length):
            # Get sequence window
            sequence_window = team_data.iloc[i:i + self.sequence_length]
            target_game = team_data.iloc[i + self.sequence_length]

            # Build sequence features
            sequence = self._build_sequence_features(sequence_window)

            # Build static features
            feature = self._build_static_features(target_game)

            # Get targets
            target = [
                target_game['Home-Team-Win'],
                target_game['OU-Cover']
            ]

            sequences.append(sequence)
            features.append(feature)
            targets.append(target)

        return {
            'sequences': sequences,
            'features': features,
            'targets': targets
        }

    def _build_sequence_features(
            self,
            sequence_data: pd.DataFrame
    ) -> np.ndarray:
        """Build features for a single sequence"""
        # Basic stats sequence
        basic_stats = sequence_data[self.config.TEAM_STATS].values

        # Form metrics sequence
        form_metrics = np.column_stack([
            sequence_data['PTS'],
            sequence_data['FG_PCT'],
            sequence_data['PLUS_MINUS']
        ])

        # Rankings sequence
        rankings = sequence_data[self.config.RANK_FEATURES].values

        return np.concatenate([
            basic_stats,
            form_metrics,
            rankings
        ], axis=1)

    def _build_static_features(
            self,
            game_data: pd.Series
    ) -> np.ndarray:
        """Build static features for a game"""
        # Team stats
        team_stats = game_data[self.config.TEAM_STATS].values

        # Rankings
        rankings = game_data[self.config.RANK_FEATURES].values

        # Rest days
        rest_days = game_data[['Days-Rest-Home', 'Days-Rest-Away']].values

        return np.concatenate([
            team_stats,
            rankings,
            rest_days
        ])
# Example usage:
"""
# Initialize
config = Config()
sequence_builder = SequenceBuilder(config)

# Get team games
home_games = get_team_games(home_team_id)
away_games = get_team_games(away_team_id)
game_date = datetime.now()

# Build sequences
game_sequences = sequence_builder.build_game_sequences(
    home_games,
    away_games,
    game_date
)

# Get matchup features
matchup_features = sequence_builder.get_matchup_features(
    game_sequences['home'],
    game_sequences['away']
)

# Prepare full feature set for model
model_features = {
    **game_sequences,
    **matchup_features
}
"""
