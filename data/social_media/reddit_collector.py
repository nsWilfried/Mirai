# data/social_media/reddit_collector.py
import numpy as np
import praw
from datetime import datetime
from typing import Dict
from textblob import TextBlob

from config.config import Config


class RedditSentimentCollector:
    def __init__(self, config: Config):
        self.reddit = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent="NBA_Prediction_Bot/1.0"
        )

    def get_team_sentiment(
            self,
            team_name: str,
            game_date: datetime
    ) -> Dict[str, float]:
        subreddits = ['nba', 'nbadiscussion']
        sentiments = []

        for subreddit_name in subreddits:
            subreddit = self.reddit.subreddit(subreddit_name)

            # Search recent posts
            for submission in subreddit.search(
                    team_name,
                    time_filter='day',
                    sort='relevance'
            ):
                blob = TextBlob(submission.title + " " + (submission.selftext or ""))
                sentiments.append(blob.sentiment.polarity)

                # Analyze top comments
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list()[:10]:
                    blob = TextBlob(comment.body)
                    sentiments.append(blob.sentiment.polarity)

        if not sentiments:
            return {
                'sentiment_score': 0,
                'sentiment_magnitude': 0,
                'post_count': 0
            }

        return {
            'sentiment_score': np.mean(sentiments),
            'sentiment_magnitude': np.std(sentiments),
            'post_count': len(sentiments)
        }
