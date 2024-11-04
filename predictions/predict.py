from typing import Any, Dict

import numpy as np
import torch

from models.model_manager import ModelManager


class ModelPredictor:
    def __init__(self, config: Any):
        self.config = config
        self.model_manager = ModelManager(config)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load all best models
        self.models = self._load_all_best_models()

    def _load_all_best_models(self) -> Dict[str, torch.nn.Module]:
        """Load all best models"""
        models = {}
        for model_type in ['lstm', 'attention', 'neural_net', 'monte_carlo']:
            try:
                model, metadata = self.model_manager.load_best_model(
                    model_type,
                    self.device
                )
                models[model_type] = {
                    'model': model,
                    'metadata': metadata
                }
                print(f"Loaded {model_type} model with metrics: {metadata['metrics']}")
            except Exception as e:
                print(f"Could not load {model_type} model: {str(e)}")

        return models

    def predict(
            self,
            features: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Make predictions using all models"""
        predictions = {}

        for model_type, model_info in self.models.items():
            model = model_info['model']
            with torch.no_grad():
                pred = model(features)
                predictions[model_type] = {
                    'prediction': pred.cpu().numpy(),
                    'metrics': model_info['metadata']['metrics']
                }

        # Calculate ensemble prediction
        ensemble_pred = self._calculate_ensemble_prediction(predictions)
        predictions['ensemble'] = ensemble_pred

        return predictions

    def _calculate_ensemble_prediction(
            self,
            predictions: Dict[str, Dict]
    ) -> Dict[str, np.ndarray]:
        """Calculate weighted ensemble prediction"""
        # Get predictions and weights
        model_preds = []
        weights = []

        for model_type, pred_info in predictions.items():
            if model_type != 'ensemble':
                model_preds.append(pred_info['prediction'])
                # Use inverse of validation loss as weight
                weights.append(1.0 / pred_info['metrics']['val_loss'])

        # Normalize weights
        weights = np.array(weights) / np.sum(weights)

        # Calculate weighted average
        ensemble_pred = np.average(
            model_preds,
            weights=weights,
            axis=0
        )

        return {
            'prediction': ensemble_pred,
            'weights': dict(zip(predictions.keys(), weights))
        }
