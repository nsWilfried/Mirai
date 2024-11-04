import numpy as np
import tweepy
from datetime import datetime, timedelta
from typing import Dict
from textblob import TextBlob

from config.config import Config


class TwitterSentimentCollector:
    def __init__(self, config: Config):
        auth = tweepy.OAuthHandler(
            config.TWITTER_API_KEY,
            config.TWITTER_API_SECRET
        )
        self.api = tweepy.API(auth)
        self.client = tweepy.Client(bearer_token=config.TWITTER_BEARER_TOKEN)

    def get_team_sentiment(
            self,
            team_name: str,
            game_date: datetime
    ) -> Dict[str, float]:
        # Search recent tweets about the team
        query = f'"{team_name}" (nba OR basketball) -is:retweet'
        start_time = game_date - timedelta(hours=24)

        tweets = self.client.search_recent_tweets(
            query=query,
            start_time=start_time,
            max_results=100
        )

        if not tweets.data:
            return {
                'sentiment_score': 0,
                'sentiment_magnitude': 0,
                'tweet_count': 0
            }

        sentiments = []
        for tweet in tweets.data:
            blob = TextBlob(tweet.text)
            sentiments.append(blob.sentiment.polarity)

        return {
            'sentiment_score': np.mean(sentiments),
            'sentiment_magnitude': np.std(sentiments),
            'tweet_count': len(sentiments)
        }
