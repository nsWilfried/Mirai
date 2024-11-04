# Find value bets
from typing import Dict, List

from config.config import Config


class ValueBetFinder:
    def __init__(self, config: Config):
        self.config = config

    def find_value_bets(
            self,
            predicted_prob: float,
            odds_dict: Dict[str, Dict[str, float]]
    ) -> List[Dict]:
        """Find value bets across all bookmakers"""
        value_bets = []

        for source, odds in odds_dict.items():
            implied_prob = 1 / odds['home_win']

            # Calculate edge
            edge = (predicted_prob - implied_prob) * 100

            if edge > self.config.MIN_VALUE_THRESHOLD:
                value_bets.append({
                    'bookmaker': source,
                    'odds': odds['home_win'],
                    'pred_prob': predicted_prob,
                    'edge': edge,
                    'rating': self._calculate_rating(edge, odds['home_win'])
                })

        return sorted(value_bets, key=lambda x: x['rating'], reverse=True)

    def _calculate_rating(self, edge: float, odds: float) -> float:
        """Calculate rating for value bet"""
        if odds < self.config.MIN_ODDS_VALUE:
            return 0
        if odds > self.config.MAX_ODDS_VALUE:
            return 0

        return edge * (1 / odds)  # Weight edge by odds
