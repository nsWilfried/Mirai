# train_models.py
import pandas as pd
import torch
from pathlib import Path
import yaml
import json
from datetime import datetime
from typing import Dict, Any, Tuple
import logging
import argparse
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from data.database import DatabaseManager
from models.attention.attention_model import MultiHeadAttentionModel
from models.lstm.lstm_model import TeamLSTM
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.sequence_builder import SequenceBuilder
from models.neural_net.deep_net import DeepNeuralNet
from models.monte_carlo.mc_simulator import MonteCarloSimulator
from training.datasets.sequence_dataset import SequenceDataset
from training.datasets.standard_dataset import StandardDataset
from training.trainers.lstm_trainer import LSTMTrainer
from training.trainers.attention_trainer import AttentionTrainer
from training.trainers.neural_net_trainer import NeuralNetTrainer
from training.trainers.monte_carlo_trainer import MonteCarloTrainer

class ModelTrainingPipeline:
    def __init__(self, config_path: str):
        # Load configuration
        self.config = self._load_config(config_path)
        # Create directories first
        self.save_dir = Path(self.config['save_dir'])
        self.models_dir = self.save_dir / 'models'
        self.logs_dir = self.save_dir / 'logs'
        self.results_dir = self.save_dir / 'results'
        self.plots_dir = self.save_dir / 'plots'

        # Create all directories
        for directory in [self.save_dir, self.models_dir, self.logs_dir,
                          self.results_dir, self.plots_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        # Setup logging
        self.logger = self._setup_logging()

        # Initialize components
        self.db_manager = DatabaseManager(self.config['database_path'])
        self.feature_engineer = FeatureEngineer(self.config)
        self.sequence_builder = SequenceBuilder(self.config)

        # Create save directories
        self.save_dir = Path(self.config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.logger.info(f"Using device: {self.device}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Ensure required config keys exist
            required_keys = ['save_dir', 'database_path']
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                raise ValueError(f"Missing required configuration keys: {missing_keys}")

            return config

        except Exception as e:
            raise ValueError(f"Error loading config file: {str(e)}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('ModelTraining')
        logger.setLevel(logging.INFO)

        # Create formatters and handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # File handler
        log_file = self.logs_dir / f'training_{datetime.now():%Y%m%d_%H%M%S}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def prepare_data(self):
        """Prepare datasets for all models"""
        self.logger.info("Loading and preparing data...")

        # Load data from database
        data = self.db_manager.get_training_data(
            self.config['data_start_date'],
            self.config['data_end_date']
        )

        # Create train/val/test splits
        train_data, val_data, test_data = self._create_data_splits(data)

        # Prepare features
        train_features = self.feature_engineer.prepare_features(train_data)
        val_features = self.feature_engineer.prepare_features(val_data)
        test_features = self.feature_engineer.prepare_features(test_data)

        # Create sequences for LSTM/Attention models
        train_sequences = self.sequence_builder.build_sequences(train_data)
        val_sequences = self.sequence_builder.build_sequences(val_data)
        test_sequences = self.sequence_builder.build_sequences(test_data)

        self.data = {
            'train': {
                'features': train_features,
                'sequences': train_sequences,
                'targets': train_data[self.config['target_column']]
            },
            'val': {
                'features': val_features,
                'sequences': val_sequences,
                'targets': val_data[self.config['target_column']]
            },
            'test': {
                'features': test_features,
                'sequences': test_sequences,
                'targets': test_data[self.config['target_column']]
            }
        }

        self.logger.info("Data preparation completed")

    def _create_data_splits(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/val/test splits"""
        train_size = self.config['train_size']
        val_size = self.config['val_size']

        # Sort by date
        data = data.sort_values('Date')

        # Calculate split indices
        train_idx = int(len(data) * train_size)
        val_idx = int(len(data) * (train_size + val_size))

        return (
            data.iloc[:train_idx],
            data.iloc[train_idx:val_idx],
            data.iloc[val_idx:]
        )

    def train_all_models(self):
        """Train all models"""
        self.logger.info("Starting training for all models...")

        results = {}
        models = {
            'lstm': (TeamLSTM, LSTMTrainer),
            'attention': (MultiHeadAttentionModel, AttentionTrainer),
            'neural_net': (DeepNeuralNet, NeuralNetTrainer),
            'monte_carlo': (MonteCarloSimulator, MonteCarloTrainer)
        }

        for model_name, (model_class, trainer_class) in models.items():
            self.logger.info(f"\nTraining {model_name} model...")
            try:
                # Initialize model
                model = model_class(self.config).to(self.device)

                # Initialize trainer
                trainer = trainer_class(
                    model=model,
                    config=self.config,
                    train_loader=self._get_data_loader('train', model_name),
                    val_loader=self._get_data_loader('val', model_name),
                    test_loader=self._get_data_loader('test', model_name)
                )

                # Train model
                model_results = trainer.train()

                # Save results
                results[model_name] = model_results
                self._save_model_results(model_name, model_results)

                self.logger.info(f"{model_name} training completed successfully")

            except Exception as e:
                self.logger.error(f"Error training {model_name} model: {str(e)}")
                continue

        # Save overall results
        self._save_overall_results(results)

        return results

    def _get_data_loader(
            self,
            split: str,
            model_name: str
    ) -> DataLoader:
        """Get appropriate data loader for model"""
        data = self.data[split]

        # Create dataset based on model type
        if model_name in ['lstm', 'attention']:
            dataset = SequenceDataset(
                sequences=data['sequences'],
                features=data['features'],
                targets=data['targets']
            )
        else:
            dataset = StandardDataset(
                features=data['features'],
                targets=data['targets']
            )

        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=(split == 'train'),
            num_workers=self.config['num_workers']
        )

    def _save_model_results(
            self,
            model_name: str,
            results: Dict[str, Any]
    ):
        """Save individual model results"""
        save_path = self.save_dir / f'{model_name}_results.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)

    def _save_overall_results(self, results: Dict[str, Any]):
        """Save overall training results"""
        # Create summary
        summary = {
            'training_date': datetime.now().isoformat(),
            'config': self.config,
            'model_performance': {}
        }

        for model_name, model_results in results.items():
            summary['model_performance'][model_name] = {
                'best_val_metrics': model_results['val_metrics'],
                'test_metrics': model_results['test_metrics']
            }

        # Save summary
        summary_path = self.save_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        self.logger.info(f"Results saved to {self.save_dir}")

    def plot_training_results(self, results: Dict[str, Any]):
        """Plot training results for all models"""
        for model_name, model_results in results.items():
            try:
                plt.figure(figsize=(12, 6))

                # Plot training and validation loss
                plt.subplot(1, 2, 1)
                plt.plot(
                    model_results['training_history']['loss'],
                    label='Train Loss'
                )
                plt.plot(
                    model_results['training_history']['val_loss'],
                    label='Val Loss'
                )
                plt.title(f'{model_name} - Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

                # Plot accuracy
                plt.subplot(1, 2, 2)
                plt.plot(
                    model_results['training_history']['accuracy'],
                    label='Train Accuracy'
                )
                plt.plot(
                    model_results['training_history']['val_accuracy'],
                    label='Val Accuracy'
                )
                plt.title(f'{model_name} - Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()

                plt.tight_layout()
                plt.savefig(self.save_dir / f'{model_name}_training.png')
                plt.close()

            except Exception as e:
                self.logger.error(f"Error plotting results for {model_name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Train NBA prediction models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ModelTrainingPipeline(args.config)

    # Prepare data
    pipeline.prepare_data()

    # Train all models
    results = pipeline.train_all_models()

    # Plot results
    pipeline.plot_training_results(results)

if __name__ == "__main__":
    main()

# Example config.yaml:
"""
# Training Configuration
database_path: "path/to/database.sqlite"
save_dir: "models/saved"
data_start_date: "2022-01-01"
data_end_date: "2023-12-31"

# Data Configuration
train_size: 0.7
val_size: 0.15
target_column: "Home-Team-Win"
batch_size: 32
num_workers: 4

# Model Parameters
lstm_params:
  hidden_size: 128
  num_layers: 2
  dropout: 0.3

attention_params:
  num_heads: 8
  hidden_size: 256
  dropout: 0.2

neural_net_params:
  hidden_sizes: [256, 128, 64]
  dropout: 0.3

monte_carlo_params:
  num_samples: 100
  dropout: 0.3

# Training Parameters
learning_rate: 0.001
weight_decay: 0.0001
epochs: 100
early_stopping_patience: 10
grad_clip: 1.0
"""

# Usage:
"""
# Create config file
config.yaml

# Run training
python train_models.py --config config.yaml
"""

# This script provides:
#
# 1. Complete Training Pipeline:
# - Data preparation
# - Model training
# - Result saving
# - Visualization
#
# 2. Features:
# - Configurable via YAML
# - Comprehensive logging
# - Error handling
# - Progress tracking
# - Result visualization
#
# 3. Output:
# - Trained models
# - Training logs
# - Performance metrics
# - Training plots
# - Summary report
#
# To use:

# config.yaml
# database_path: "nba_data.sqlite"
# save_dir: "saved_models"
# data_start_date: "2022-01-01"
# data_end_date: "2023-12-31"
# ... other parameters

# 2. Run training:
# ```bash
# python train_models.py --config config.yaml
# ```
#
# 3. Check results:
# ```python
# Load results
# with open("saved_models/training_summary.json", 'r') as f:
#     results = json.load(f)
#
# # Print performance
# for model, metrics in results['model_performance'].items():
#     print(f"\n{model} Performance:")
#     print(f"Validation: {metrics['best_val_metrics']}")
#     print(f"Test: {metrics['test_metrics']}")
