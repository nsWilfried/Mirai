# Calculate rest days
from datetime import datetime, timedelta
import pandas as pd

class RestCalculator:
    @staticmethod
    def calculate_rest_days(
            team_id: int,
            game_date: datetime,
            recent_games: pd.DataFrame
    ) -> int:
        """Calculate days of rest for a team"""
        if recent_games.empty:
            return 3  # Default to 3 days if no recent games

        last_game_date = pd.to_datetime(
            recent_games.iloc[0]['game_date']
        )

        rest_days = (game_date - last_game_date).days
        return min(rest_days, 5)  # Cap at 5 days
