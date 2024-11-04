# config.py
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Config:
    # Database configuration
    DATABASE_PATH: str = "/data/sports/nba/dataset.sqlite"
    # Social Media API Keys
    TWITTER_API_KEY: str = "JqUm8lt48CD7sZylY0glvWmlU"
    TWITTER_API_SECRET: str = "7D3PmvgRiRW77u6g6ZBzfWUKXgywQSmsNxe48qWoQ9KOGLnjn5"
    TWITTER_ACCESS_TOKEN: str = "1765403224780013568-w4jPRVIR9FGNhtnGiXWfBHl5wjk9el"
    TWITTER_BEARER_TOKEN: str = "AAAAAAAAAAAAAAAAAAAAABUHwwEAAAAACBqZ%2BGU%2Fl1i5RW8DhOn20ZAS%2BzM%3D5M76OPayOLCCTZ2udMbwqyjA0fY78oi6tptQ0Ei6JM40pGGq4u"
    MIN_ODDS_VALUE: float = 1.5
    MAX_ODDS_VALUE: float = 10.0
    MIN_VALUE_THRESHOLD: float = 0.05  # 5% difference
    ODDS_API_KEY="9aa1ac20248799da21dcca6ba76d3b8c"
    # Basic statistics features
    BASIC_STATS: List[str] = field(default_factory=lambda: [
        'GP', 'W', 'L', 'W_PCT', 'MIN',
        'FGM', 'FGA', 'FG_PCT',
        'FG3M', 'FG3A', 'FG3_PCT',
        'FTM', 'FTA', 'FT_PCT',
        'OREB', 'DREB', 'REB',
        'AST', 'TOV', 'STL', 'BLK',
        'BLKA', 'PF', 'PFD',
        'PTS', 'PLUS_MINUS'
    ])

    # Ranking features
    RANK_FEATURES: List[str] = field(default_factory=lambda: [
        'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK',
        'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK',
        'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK',
        'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK',
        'OREB_RANK', 'DREB_RANK', 'REB_RANK',
        'AST_RANK', 'TOV_RANK', 'STL_RANK',
        'BLK_RANK', 'BLKA_RANK', 'PF_RANK',
        'PFD_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK'
    ])

    # Key performance indicators
    KEY_STATS: List[str] = field(default_factory=lambda: [
        'W_PCT', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
        'REB', 'AST', 'TOV', 'PTS', 'PLUS_MINUS'
    ])

    # Target columns
    TARGET_COLUMNS: List[str] = field(default_factory=lambda: [
        'Home-Team-Win', 'OU-Cover'
    ])

    # Model parameters
    SEQUENCE_LENGTH: int = 10
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 10

    # Model architecture parameters
    LSTM_PARAMS: Dict = field(default_factory=lambda: {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": True
    })

    ATTENTION_PARAMS: Dict = field(default_factory=lambda: {
        "num_heads": 8,
        "hidden_size": 256,
        "num_layers": 3,
        "dropout": 0.2
    })

    NEURAL_NET_PARAMS: Dict = field(default_factory=lambda: {
        "hidden_sizes": [256, 128, 64],
        "dropout": 0.3,
        "activation": "relu"
    })

    MONTE_CARLO_PARAMS: Dict = field(default_factory=lambda: {
        "num_simulations": 1000,
        "confidence_threshold": 0.6
    })

    BOOKMAKERS: List[str] = field(default_factory=lambda: [
        "draftkings",
        "fanduel",
        "bovada",
        "betmgm",
        "caesars",
        "pointsbetus"
    ])

    # Feature engineering parameters
    ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [3, 5, 10])
    MAX_REST_DAYS: int = 5

    # Model weights for ensemble
    MODEL_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "lstm": 0.3,
        "attention": 0.25,
        "neural_net": 0.25,
        "monte_carlo": 0.2
    })
