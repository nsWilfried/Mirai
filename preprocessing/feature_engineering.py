# preprocessing/feature_engineering.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime

class FeatureEngineer:
    def __init__(self, config: Any):
        self.config = config
        self.scalers = {
            'team_stats': StandardScaler(),
            'rankings': MinMaxScaler(),
            'advanced': StandardScaler(),
            'form': StandardScaler()
        }
        self.fitted = False

    def prepare_features(
            self,
            data: pd.DataFrame,
            fit: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Prepare all features from data

        Args:
            data: DataFrame containing raw data
            fit: Whether to fit the scalers (only for training data)
        """
        # Basic team statistics
        team_stats = self._prepare_team_stats(data)

        # Ranking features
        rankings = self._prepare_rankings(data)

        # Form and momentum features
        form = self._prepare_form_features(data)

        # Advanced metrics
        advanced = self._calculate_advanced_metrics(data)

        # Scale features
        if fit and not self.fitted:
            self._fit_scalers(team_stats, rankings, form, advanced)
            self.fitted = True

        features = {
            'team_stats': self.scalers['team_stats'].transform(team_stats),
            'rankings': self.scalers['rankings'].transform(rankings),
            'form': self.scalers['form'].transform(form),
            'advanced': self.scalers['advanced'].transform(advanced)
        }

        return features

    def _prepare_team_stats(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare basic team statistics"""
        team_stats = data[self.config.TEAM_STATS].values
        return np.nan_to_num(team_stats, nan=0)

    def _prepare_rankings(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare ranking features"""
        rankings = data[self.config.RANK_FEATURES].values
        # Normalize rankings to [0,1] range
        return rankings / 30.0  # Assuming 30 teams

    def _prepare_form_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate form and momentum features"""
        form_features = []

        for _, team_data in data.groupby('TEAM_NAME'):
            team_data = team_data.sort_values('Date')

            # Calculate streaks
            wins = (team_data['Home-Team-Win'] == 1).astype(int)
            streak = self._calculate_streak(wins)

            # Calculate moving averages
            pts_ma = team_data['PTS'].rolling(5).mean()
            fg_pct_ma = team_data['FG_PCT'].rolling(5).mean()
            plus_minus_ma = team_data['PLUS_MINUS'].rolling(5).mean()

            # Calculate momentum
            pts_momentum = team_data['PTS'].diff().rolling(3).mean()
            plus_minus_momentum = team_data['PLUS_MINUS'].diff().rolling(3).mean()

            form_features.append(np.column_stack([
                streak,
                pts_ma,
                fg_pct_ma,
                plus_minus_ma,
                pts_momentum,
                plus_minus_momentum
            ]))

        return np.vstack(form_features)

    def _calculate_advanced_metrics(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate advanced basketball metrics"""
        # Possessions
        poss = (data['FGA'] + 0.4 * data['FTA'] - 1.07 *
                (data['OREB'] / (data['OREB'] + data['DREB'])) *
                (data['FGA'] - data['FGM']) + data['TOV'])

        # Offensive Rating
        off_rating = data['PTS'] / poss * 100

        # True Shooting Percentage
        ts_pct = data['PTS'] / (2 * (data['FGA'] + 0.44 * data['FTA']))

        # Assist Ratio
        ast_ratio = data['AST'] / (data['FGA'] + 0.44 * data['FTA'] + data['AST'] + data['TOV'])

        # Turnover Ratio
        tov_ratio = data['TOV'] / (data['FGA'] + 0.44 * data['FTA'] + data['TOV'])

        return np.column_stack([
            poss,
            off_rating,
            ts_pct,
            ast_ratio,
            tov_ratio
        ])

    def _fit_scalers(
            self,
            team_stats: np.ndarray,
            rankings: np.ndarray,
            form: np.ndarray,
            advanced: np.ndarray
    ):
        """Fit all scalers"""
        self.scalers['team_stats'].fit(team_stats)
        self.scalers['rankings'].fit(rankings)
        self.scalers['form'].fit(form)
        self.scalers['advanced'].fit(advanced)

    def _calculate_streak(self, results: pd.Series) -> int:
        """Calculate current streak"""
        streak = 0
        for result in results[::-1]:
            if streak == 0:
                streak = 1 if result else -1
            elif (streak > 0 and result) or (streak < 0 and not result):
                streak += 1 if result else -1
            else:
                break
        return streak
