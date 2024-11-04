from config.config import Config

# data/odds_scraper.py
import requests
from datetime import datetime
from typing import Dict, List, Optional
import time
from dataclasses import dataclass

@dataclass
class OddsResponse:
    """Structured response for odds data"""
    home_team: str
    away_team: str
    game_id: str
    commence_time: datetime
    bookmakers: List[Dict]
    best_home_odds: float
    best_away_odds: float
    best_over_odds: Optional[float] = None
    best_under_odds: Optional[float] = None
    home_probabilities: List[float] = None
    away_probabilities: List[float] = None

class OddsScraper:
    def __init__(self, config: Config):
        self.api_key = config.ODDS_API_KEY
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.sport = "basketball_nba"
        self.regions = "us"  # eu, uk, us, au
        self.markets = ["h2h", "totals"]  # h2h for moneyline, totals for over/under
        self.bookmakers = config.BOOKMAKERS  # List of bookmakers to track
        self.odds_format = "decimal"

        # Rate limiting
        self.requests_remaining = None
        self.last_request_time = None
        self.min_request_interval = 1  # Minimum seconds between requests

    def get_odds_for_game(
            self,
            home_team: str,
            away_team: str,
            game_date: datetime
    ) -> Optional[OddsResponse]:
        """Get odds for a specific game"""
        try:
            # Respect rate limiting
            self._handle_rate_limiting()

            # Get moneyline odds
            h2h_odds = self._fetch_odds("h2h", home_team, away_team)

            # Get totals (over/under) odds
            totals_odds = self._fetch_odds("totals", home_team, away_team)

            if not h2h_odds:
                return None

            # Process and combine odds
            odds_response = self._process_odds(h2h_odds, totals_odds, home_team, away_team)

            return odds_response

        except Exception as e:
            print(f"Error fetching odds: {str(e)}")
            return None

    def get_all_games_odds(self) -> List[OddsResponse]:
        """Get odds for all upcoming NBA games"""
        try:
            # Respect rate limiting
            self._handle_rate_limiting()

            # Get all games moneyline odds
            h2h_odds = self._fetch_all_odds("h2h")

            # Get all games totals odds
            totals_odds = self._fetch_all_odds("totals")

            # Process odds for each game
            all_odds = []
            for game in h2h_odds:
                game_totals = next(
                    (t for t in totals_odds if t['id'] == game['id']),
                    None
                )

                odds_response = self._process_odds(
                    game,
                    game_totals,
                    game['home_team'],
                    game['away_team']
                )

                if odds_response:
                    all_odds.append(odds_response)

            return all_odds

        except Exception as e:
            print(f"Error fetching all odds: {str(e)}")
            return []

    def _fetch_odds(
            self,
            market: str,
            home_team: str,
            away_team: str
    ) -> Optional[Dict]:
        """Fetch odds for specific market and game"""
        url = f"{self.base_url}/{self.sport}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': self.regions,
            'markets': market,
            'oddsFormat': self.odds_format
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            self._update_rate_limits(response.headers)

            # Find the specific game
            games = response.json()
            for game in games:
                if (game['home_team'] == home_team and
                        game['away_team'] == away_team):
                    return game

            return None
        else:
            raise Exception(f"API request failed: {response.status_code}")

    def _fetch_all_odds(self, market: str) -> List[Dict]:
        """Fetch odds for all games for specific market"""
        url = f"{self.base_url}/{self.sport}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': self.regions,
            'markets': market,
            'oddsFormat': self.odds_format
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            self._update_rate_limits(response.headers)
            return response.json()
        else:
            raise Exception(f"API request failed: {response.status_code}")

    def _process_odds(
            self,
            h2h_data: Dict,
            totals_data: Optional[Dict],
            home_team: str,
            away_team: str
    ) -> OddsResponse:
        """Process odds data into structured format"""
        # Process moneyline odds
        home_odds = []
        away_odds = []
        bookmaker_data = []

        for bookmaker in h2h_data['bookmakers']:
            if bookmaker['key'] in self.bookmakers:
                for market in bookmaker['markets']:
                    if market['key'] == 'h2h':
                        # Find home and away odds
                        for outcome in market['outcomes']:
                            if outcome['name'] == home_team:
                                home_odds.append(outcome['price'])
                            elif outcome['name'] == away_team:
                                away_odds.append(outcome['price'])

                        bookmaker_data.append({
                            'name': bookmaker['key'],
                            'home_odds': home_odds[-1],
                            'away_odds': away_odds[-1]
                        })

        # Process totals odds if available
        over_odds = None
        under_odds = None
        if totals_data:
            over_prices = []
            under_prices = []
            for bookmaker in totals_data['bookmakers']:
                if bookmaker['key'] in self.bookmakers:
                    for market in bookmaker['markets']:
                        if market['key'] == 'totals':
                            for outcome in market['outcomes']:
                                if outcome['name'] == 'Over':
                                    over_prices.append(outcome['price'])
                                elif outcome['name'] == 'Under':
                                    under_prices.append(outcome['price'])

            if over_prices and under_prices:
                over_odds = max(over_prices)
                under_odds = max(under_prices)

        # Calculate implied probabilities
        home_probabilities = [1/odd for odd in home_odds]
        away_probabilities = [1/odd for odd in away_odds]

        return OddsResponse(
            home_team=home_team,
            away_team=away_team,
            game_id=h2h_data['id'],
            commence_time=datetime.fromisoformat(h2h_data['commence_time'].replace('Z', '+00:00')),
            bookmakers=bookmaker_data,
            best_home_odds=max(home_odds) if home_odds else None,
            best_away_odds=max(away_odds) if away_odds else None,
            best_over_odds=over_odds,
            best_under_odds=under_odds,
            home_probabilities=home_probabilities,
            away_probabilities=away_probabilities
        )

    def _handle_rate_limiting(self):
        """Handle API rate limiting"""
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)

        if self.requests_remaining is not None and self.requests_remaining <= 0:
            raise Exception("API request limit reached")

    def _update_rate_limits(self, headers: Dict):
        """Update rate limiting information from response headers"""
        self.requests_remaining = int(headers.get('x-requests-remaining', 0))
        self.last_request_time = time.time()

# Example usage:
"""
Key features:

1. API Integration:
- Uses the-odds-api.com's V4 API
- Supports multiple markets (moneyline, totals)
- Rate limiting handling
- Error handling

2. Odds Collection:
- Moneyline odds
- Over/Under odds
- Best odds tracking
- Multiple bookmakers

3. Data Structure:
- Clean OddsResponse class
- Bookmaker-specific information
- Implied probabilities
- Game identification

4. Rate Limiting:
- Tracks remaining requests
- Enforces minimum request intervals
- Handles API limits gracefully

# Initialize
config = Config()
odds_scraper = OddsScraper(config)

# Get odds for specific game
odds = odds_scraper.get_odds_for_game(
    "Los Angeles Lakers",
    "Golden State Warriors",
    datetime.now()
)

if odds:
    print(f"Best Home Odds: {odds.best_home_odds}")
    print(f"Best Away Odds: {odds.best_away_odds}")
    print(f"Best Over Odds: {odds.best_over_odds}")
    print(f"Best Under Odds: {odds.best_under_odds}")
    
    print("\nBookmaker Details:")
    for bookmaker in odds.bookmakers:
        print(f"{bookmaker['name']}: Home {bookmaker['home_odds']}, "
              f"Away {bookmaker['away_odds']}")

# Get odds for all games
all_odds = odds_scraper.get_all_games_odds()
for game_odds in all_odds:
    print(f"\n{game_odds.home_team} vs {game_odds.away_team}")
    print(f"Best Odds - Home: {game_odds.best_home_odds}, "
          f"Away: {game_odds.best_away_odds}")
"""
