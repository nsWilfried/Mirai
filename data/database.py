import logging
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = self._setup_logger()
    def _setup_logger(self) -> logging.Logger:
        """Setup database logger"""
        logger = logging.getLogger('DatabaseManager')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger
    def get_upcoming_games(self, date: datetime) -> pd.DataFrame:
        """Get upcoming games for specified date"""
        query = """
        SELECT 
            g.game_id,
            g.home_team,
            g.away_team,
            g.game_date,
            g.game_time
        FROM schedules g
        WHERE DATE(g.game_date) = DATE(?)
        """

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=(date.date()))

    def get_team_recent_games(
            self,
            team_name: str,
            before_date: datetime
    ) -> pd.DataFrame:
        """
        Get recent games for a team before specified date using team name

        Args:
            team_name: Name of the team (e.g., 'Boston Celtics')
            before_date: Get games before this date

        Returns:
            DataFrame containing recent games
        """
        query = """
            SELECT *
            FROM nba_games
            WHERE (TEAM_NAME = ? OR TEAM_NAME.1 = ?)
                AND Date < ?
            ORDER BY Date DESC
            LIMIT 10
            """

        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(team_name, team_name, before_date)
                )

                if df.empty:
                    print(f"No games found for {team_name} before {before_date}")
                    return pd.DataFrame()

                # Standardize the data so team of interest is always in the main columns
                standardized_games = []
                for _, game in df.iterrows():
                    if game['TEAM_NAME.1'] == team_name:
                        # Swap columns to make team of interest the main team
                        swapped_game = self._swap_team_columns(game)
                        standardized_games.append(swapped_game)
                    else:
                        standardized_games.append(game)

                return pd.DataFrame(standardized_games)

        except sqlite3.Error as e:
            print(f"Database error: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error getting recent games: {str(e)}")
            return pd.DataFrame()


    def get_training_data(
            self,
            start_date: str,
            end_date: str
    ) -> pd.DataFrame:
        """
        Get training data between specified dates

        Args:
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'

        Returns:
            DataFrame containing game data
        """
        try:
            query = """
            SELECT *
            FROM nba_games
            WHERE Date >= ? AND Date <= ?
            ORDER BY Date ASC
            """

            self.logger.info(f"Fetching data from {start_date} to {end_date}")

            with sqlite3.connect(self.db_path) as conn:
                # Read data
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(start_date, end_date),
                    parse_dates=['Date']
                )

                if df.empty:
                    self.logger.warning("No data found for specified date range")
                    return pd.DataFrame()

                # Basic data validation
                missing_cols = set(self._get_required_columns()) - set(df.columns)
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")

                self.logger.info(f"Retrieved {len(df)} games")

                # Add derived features
                df = self._add_derived_features(df)

                # Handle missing values
                df = self._handle_missing_values(df)

                return df

        except sqlite3.Error as e:
            self.logger.error(f"Database error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error getting training data: {str(e)}")
            raise

    def _get_required_columns(self) -> List[str]:
        """List of required columns for training"""
        return [
            'TEAM_NAME', 'TEAM_NAME.1',
            'GP', 'W', 'L', 'W_PCT',
            'MIN', 'FGM', 'FGA', 'FG_PCT',
            'FG3M', 'FG3A', 'FG3_PCT',
            'FTM', 'FTA', 'FT_PCT',
            'OREB', 'DREB', 'REB',
            'AST', 'TOV', 'STL', 'BLK',
            'BLKA', 'PF', 'PFD', 'PTS',
            'PLUS_MINUS',
            # Rankings
            'GP_RANK', 'W_RANK', 'L_RANK',
            'W_PCT_RANK', 'MIN_RANK',
            'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
            'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK',
            'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK',
            'OREB_RANK', 'DREB_RANK', 'REB_RANK',
            'AST_RANK', 'TOV_RANK', 'STL_RANK',
            'BLK_RANK', 'BLKA_RANK', 'PF_RANK',
            'PFD_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK',
            # Game specific
            'Date', 'Score', 'Home-Team-Win',
            'OU', 'OU-Cover', 'Days-Rest-Home',
            'Days-Rest-Away'
        ]

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataset"""
        # Calculate point differentials
        try:
            scores = df['Score'].str.split('-', expand=True).astype(float)
            df['point_differential'] = scores[0] - scores[1]
        except:
            self.logger.warning("Could not calculate point differential")
            df['point_differential'] = 0

        # Calculate rolling averages for key stats
        for team_col in ['TEAM_NAME', 'TEAM_NAME.1']:
            for stat in ['PTS', 'PLUS_MINUS', 'FG_PCT']:
                col_name = f'{stat}_rolling_5' if team_col == 'TEAM_NAME' else f'{stat}_rolling_5_away'
                df[col_name] = df.groupby(team_col)[stat].transform(
                    lambda x: x.rolling(5, min_periods=1).mean()
                )

        # Calculate win/loss streaks
        df['home_streak'] = df.groupby('TEAM_NAME')['Home-Team-Win'].transform(
            lambda x: x.rolling(5, min_periods=1).sum()
        )
        df['away_streak'] = df.groupby('TEAM_NAME.1')['Home-Team-Win'].transform(
            lambda x: (~x).rolling(5, min_periods=1).sum()
        )

        # Add day of week
        df['day_of_week'] = df['Date'].dt.dayofweek

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill missing numerical values with appropriate strategies
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

        for col in numerical_cols:
            if col.endswith('_RANK'):
                # Fill ranking missing values with median
                df[col] = df[col].fillna(df[col].median())
            elif col in ['Days-Rest-Home', 'Days-Rest-Away']:
                # Fill rest days with 1 (most common)
                df[col] = df[col].fillna(1)
            elif col.endswith('_PCT'):
                # Fill percentages with mean
                df[col] = df[col].fillna(df[col].mean())
            else:
                # Fill other stats with 0
                df[col] = df[col].fillna(0)

        # Log missing value handling
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            self.logger.warning("Remaining missing values after handling:")
            for col, count in missing_counts[missing_counts > 0].items():
                self.logger.warning(f"{col}: {count} missing values")

        return df

    def get_test_data(
            self,
            start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get most recent games for testing

        Args:
            start_date: Optional start date, defaults to last 30 days

        Returns:
            DataFrame containing recent games
        """
        if start_date is None:
            start_date = (
                    pd.Timestamp.now() - pd.Timedelta(days=30)
            ).strftime('%Y-%m-%d')

        return self.get_training_data(
            start_date=start_date,
            end_date=pd.Timestamp.now().strftime('%Y-%m-%d')
        )

    def get_team_data(
            self,
            team_name: str,
            n_games: int = 10
    ) -> pd.DataFrame:
        """
        Get recent games for a specific team

        Args:
            team_name: Team name
            n_games: Number of recent games to retrieve

        Returns:
            DataFrame containing team's recent games
        """
        try:
            query = """
            SELECT *
            FROM nba_games
            WHERE TEAM_NAME = ? OR TEAM_NAME.1 = ?
            ORDER BY Date DESC
            LIMIT ?
            """

            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(team_name, team_name, n_games),
                    parse_dates=['Date']
                )

                if df.empty:
                    self.logger.warning(f"No games found for {team_name}")
                    return pd.DataFrame()

                # Add derived features
                df = self._add_derived_features(df)
                df = self._handle_missing_values(df)

                return df

        except Exception as e:
            self.logger.error(f"Error getting team data: {str(e)}")
            raise

    # def get_last_game_date(
    #         self,
    #         team_name: str
    # ) -> Optional[datetime]:
    #     """Get the date of the team's last game"""
    #     query = """
    #     SELECT Date
    #     FROM nba_games
    #     WHERE TEAM_NAME = ? OR TEAM_NAME.1 = ?
    #     ORDER BY Date DESC
    #     LIMIT 1
    #     """
    #
    #     try:
    #         with sqlite3.connect(self.db_path) as conn:
    #             df = pd.read_sql_query(
    #                 query,
    #                 conn,
    #                 params=(team_name, team_name)
    #             )
    #             if not df.empty:
    #                 return pd.to_datetime(df['Date'].iloc[0])
    #             return None
    #     except Exception as e:
    #         print(f"Error getting last game date: {str(e)}")
    #         return None
    def _swap_team_columns(self, game):
        pass


# Example usage:
"""
# Initialize database manager
db_manager = DatabaseManager("nba_database.sqlite")

# Get recent games for a team
celtics_games = db_manager.get_team_recent_games(
    "Boston Celtics",
    datetime.now()
)

# Print results
if not celtics_games.empty:
    print(f"Found {len(celtics_games)} recent games")
    for _, game in celtics_games.iterrows():
        print(f"{game['TEAM_NAME']} vs {game['TEAM_NAME.1']}: "
              f"{game['PTS']}-{game['PTS.1']}")

# Get last game date
last_game = db_manager.get_last_game_date("Boston Celtics")
if last_game:
    print(f"Last game was on: {last_game.date()}")
"""
