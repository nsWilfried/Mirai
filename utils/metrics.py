class PredictionMetrics:
    # @staticmethod
    # def calculate_edge(
    #         predicted_prob: float,
    #         odds: float
    # ) -> float:
    #     """Calculate betting edge"""
    #     implied_prob = 1 / odds
    #     return (predicted_prob - implied_prob) * 100

    @staticmethod
    def calculate_kelly_criterion(
            predicted_prob: float,
            odds: float
    ) -> float:
        """Calculate Kelly Criterion bet size"""
        implied_prob = 1 / odds
        edge = predicted_prob - implied_prob

        if edge <= 0:
            return 0

        return (predicted_prob * (odds - 1) - (1 - predicted_prob)) / (odds - 1)
