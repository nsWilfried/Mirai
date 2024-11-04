# models/ensemble/ensemble.py
from typing import Dict, Any

import torch

from config.config import Config
from models.attention.attention_model import MultiHeadAttentionModel
from models.lstm.lstm_model import TeamLSTM
from models.monte_carlo.mc_simulator import MonteCarloSimulator
from models.neural_net.deep_net import DeepNeuralNet


class EnsemblePredictor:
    def __init__(self, config: Config):
        self.config = config

        # Initialize all models
        self.models = {
            'lstm': TeamLSTM(config),
            'attention': MultiHeadAttentionModel(config),
            'monte_carlo': MonteCarloSimulator(config),
            'neural_net': DeepNeuralNet(config)
        }

        self.weights = config.MODEL_WEIGHTS

    def predict(
            self,
            features: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        predictions = {}
        confidences = {}

        # Get predictions from each model
        for name, model in self.models.items():
            pred, conf = model.predict(features)
            predictions[name] = pred
            confidences[name] = conf

        # Calculate weighted ensemble prediction
        weighted_pred = sum(
            predictions[name] * self.weights[name]
            for name in predictions
        )

        # Calculate ensemble confidence
        weighted_conf = sum(
            confidences[name] * self.weights[name]
            for name in confidences
        )

        return {
            'ensemble_prediction': weighted_pred,
            'ensemble_confidence': weighted_conf,
            'model_predictions': predictions,
            'model_confidences': confidences
        }
