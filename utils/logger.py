# utils/logger.py
import logging
from datetime import datetime

from config.config import Config


class Logger:
    def __init__(self, config: Config):
        logging.basicConfig(
            filename=f'logs/nba_prediction_{datetime.now().date()}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('NBA_Prediction')

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(message)

    def warning(self, message: str):
        self.logger.warning(message)
