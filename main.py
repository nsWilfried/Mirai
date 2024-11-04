# main.py
import argparse
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import torch

from data.database import DatabaseManager
from data.odds_scraper import OddsScraper
from data.social_media.twitter_collector import TwitterSentimentCollector
from data.social_media.reddit_collector import RedditCollector
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.sequence_builder import SequenceBuilder

class NBAPredictionSystem:
    def __init__(self, config_path: str):
        # Load configuration
        self.config = self._load_config(config_path)

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize components
        self.db_manager = DatabaseManager(self.config['database_path'])
        self.odds_scraper = OddsScraper(self.config)
        self.twitter_collector = TwitterSentimentCollector(self.config)
        self.reddit_collector = RedditCollector(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.sequence_builder = SequenceBuilder(self.config)

        # Load models
        self.models = self._load_models()

        # Create output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return json.load(f)

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('NBAPrediction')
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(
            self.output_dir / f'predictions_{datetime.now():%Y%m%d}.log'
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _load_models(self) -> Dict[str, Any]:
        """Load all trained models"""
        models = {}
        model_dir = Path(self.config['model_dir'])

        for model_type in ['lstm', 'attention', 'neural_net', 'monte_carlo']:
            try:
                model_path = model_dir / f'{model_type}_final.pth'
                models[model_type] = torch.load(
                    model_path,
                    map_location=self.device
                )
                self.logger.info(f"Loaded {model_type} model")
            except Exception as e:
                self.logger.error(f"Error loading {model_type} model: {str(e)}")

        return models

    def get_predictions(self) -> Dict[str, Any]:
        """Get predictions for all upcoming games"""
        try:
            # Get upcoming games from odds API
            self.logger.info("Fetching upcoming games odds...")
            games = self.odds_scraper.get_all_games_odds()

            if not games:
                self.logger.info("No upcoming games found")
                return {}

            self.logger.info(f"Found {len(games)} upcoming games")

            # Process each game
            with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                predictions = list(executor.map(
                    self._process_game,
                    games
                ))

            # Combine and save predictions
            results = self._combine_predictions(games, predictions)
            self._save_predictions(results)

            return results

        except Exception as e:
            self.logger.error(f"Error in prediction process: {str(e)}")
            return {}

    def _process_game(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single game"""
        try:
            # Get teams' recent games
            home_data = self.db_manager.get_team_data(
                game['home_team'],
                n_games=self.config['n_recent_games']
            )
            away_data = self.db_manager.get_team_data(
                game['away_team'],
                n_games=self.config['n_recent_games']
            )

            # Get sentiment data
            sentiment = self._get_sentiment_data(
                game['home_team'],
                game['away_team'],
                game['commence_time']
            )

            # Prepare features
            features = self._prepare_features(
                home_data,
                away_data,
                sentiment,
                game['commence_time']
            )

            # Get predictions from all models
            predictions = self._get_model_predictions(features)

            # Calculate value bets
            value_bets = self._analyze_value_bets(
                predictions,
                game['bookmakers']
            )

            return {
                'predictions': predictions,
                'sentiment': sentiment,
                'value_bets': value_bets
            }

        except Exception as e:
            self.logger.error(f"Error processing game: {str(e)}")
            return None

    def _get_sentiment_data(
            self,
            home_team: str,
            away_team: str,
            game_time: datetime
    ) -> Dict[str, Any]:
        """Get sentiment data for both teams"""
        sentiment = {
            'home': {
                'twitter': self.twitter_collector.get_team_sentiment(
                    home_team,
                    game_time
                ),
                'reddit': self.reddit_collector.get_team_sentiment(
                    home_team,
                    game_time
                )
            },
            'away': {
                'twitter': self.twitter_collector.get_team_sentiment(
                    away_team,
                    game_time
                ),
                'reddit': self.reddit_collector.get_team_sentiment(
                    away_team,
                    game_time
                )
            }
        }

        # Calculate combined sentiment scores
        for team in ['home', 'away']:
            sentiment[team]['combined'] = (
                    sentiment[team]['twitter']['sentiment_score'] * 0.6 +
                    sentiment[team]['reddit']['sentiment_score'] * 0.4
            )

        return sentiment

    def _prepare_features(
            self,
            home_data: pd.DataFrame,
            away_data: pd.DataFrame,
            sentiment: Dict[str, Any],
            game_time: datetime
    ) -> Dict[str, torch.Tensor]:
        """Prepare features for prediction"""
        # Basic features
        features = self.feature_engineer.prepare_features(
            pd.concat([home_data, away_data])
        )

        # Sequence features
        sequences = self.sequence_builder.build_sequences(
            pd.concat([home_data, away_data])
        )

        # Add sentiment features
        sentiment_features = torch.tensor([
            sentiment['home']['combined'],
            sentiment['away']['combined']
        ], dtype=torch.float32).unsqueeze(0)

        return {
            'features': features,
            'sequences': sequences,
            'sentiment': sentiment_features
        }

    def _get_model_predictions(
            self,
            features: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Get predictions from all models"""
        predictions = {}

        for model_name, model in self.models.items():
            try:
                with torch.no_grad():
                    model.eval()
                    if model_name in ['lstm', 'attention']:
                        output = model({
                            'sequence': features['sequences'],
                            'features': features['features'],
                            'sentiment': features['sentiment']
                        })
                    else:
                        output = model({
                            'features': features['features'],
                            'sentiment': features['sentiment']
                        })

                    predictions[model_name] = {
                        'win_prob': float(output[0].item()),
                        'ou_prob': float(output[1].item())
                    }
            except Exception as e:
                self.logger.error(f"Error with {model_name} prediction: {str(e)}")
                predictions[model_name] = None

        # Calculate ensemble prediction
        valid_predictions = [
            pred['win_prob']
            for pred in predictions.values()
            if pred is not None
        ]

        if valid_predictions:
            predictions['ensemble'] = {
                'win_prob': np.mean(valid_predictions),
                'confidence': np.std(valid_predictions)
            }

        return predictions

    def _analyze_value_bets(
            self,
            predictions: Dict[str, Dict[str, float]],
            odds: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze value betting opportunities"""
        value_bets = []

        if 'ensemble' not in predictions:
            return value_bets

        pred_prob = predictions['ensemble']['win_prob']
        confidence = predictions['ensemble']['confidence']

        for book in odds:
            try:
                # Convert odds to probability
                implied_prob = 1 / book['home_odds']

                # Calculate edge
                edge = (pred_prob - implied_prob) * 100

                # Check for value opportunities
                if abs(edge) > self.config['min_edge'] and confidence < self.config['max_uncertainty']:
                    value_bets.append({
                        'bookmaker': book['bookmaker'],
                        'odds': book['home_odds'],
                        'edge': edge,
                        'confidence': confidence,
                        'bet_type': 'home' if edge > 0 else 'away'
                    })
            except Exception as e:
                self.logger.error(f"Error analyzing odds from {book['bookmaker']}: {str(e)}")

        return sorted(value_bets, key=lambda x: abs(x['edge']), reverse=True)

    def _combine_predictions(
            self,
            games: List[Dict[str, Any]],
            predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine games and predictions"""
        results = []

        for game, pred in zip(games, predictions):
            if pred is None:
                continue

            results.append({
                'game_info': {
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'commence_time': game['commence_time'].isoformat(),
                    'best_home_odds': game['best_home_odds'],
                    'best_away_odds': game['best_away_odds']
                },
                'predictions': pred['predictions'],
                'sentiment': pred['sentiment'],
                'value_bets': pred['value_bets']
            })

        return {
            'timestamp': datetime.now().isoformat(),
            'predictions': results
        }

    def _save_predictions(self, results: Dict[str, Any]):
        """Save predictions to file"""
        output_file = self.output_dir / f'predictions_{datetime.now():%Y%m%d_%H%M}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        self.logger.info(f"Predictions saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='NBA Game Predictions')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Initialize prediction system
    system = NBAPredictionSystem(args.config)

    # Get predictions
    predictions = system.get_predictions()

    # Print results
    if predictions:
        print("\nPredictions Summary:")
        for game in predictions['predictions']:
            print(f"\n{game['game_info']['home_team']} vs {game['game_info']['away_team']}")
            print(f"Game Time: {game['game_info']['commence_time']}")
            print("\nModel Predictions:")
            for model, pred in game['predictions'].items():
                if pred:
                    print(f"{model}: {pred['win_prob']:.3f}")

            if game['value_bets']:
                print("\nValue Betting Opportunities:")
                for bet in game['value_bets']:
                    print(f"Bookmaker: {bet['bookmaker']}")
                    print(f"Odds: {bet['odds']:.2f}")
                    print(f"Edge: {bet['edge']:.1f}%")
                    print(f"Confidence: {bet['confidence']:.3f}")

if __name__ == "__main__":
    main()

# Key features:
#
# 1. Complete Integration:
# - Odds collection
# - Sentiment analysis
# - Feature preparation
# - Model predictions
# - Value bet analysis
#
# 2. Parallel Processing:
# - Thread pool for game processing
#     - Concurrent API requests
# - Efficient data collection
#
# 3. Comprehensive Analysis:
# - Multiple model predictions
# - Ensemble predictions
# - Sentiment integration
# - Value bet identification
#
# 4. Output:
# - Detailed predictions
# - Confidence scores
# - Value betting opportunities
# - Complete logging
#
# Usage:
# Create config file
# config.json
# {
#     "database_path": "path/to/database.sqlite",
#     "model_dir": "path/to/models",
#     "output_dir": "predictions",
#     "n_recent_games": 10,
#     "max_workers": 4,
#     "min_edge": 5.0,
#     "max_uncertainty": 0.2
#     # ... other settings
# }

# Run predictions
# python main.py --config config.json
#
# Would you like me to:
# 1. Add more analysis metrics?
# 2. Enhance the value betting logic?
# 3. Add visualization outputs?
# 4. Include model confidence adjustments?
