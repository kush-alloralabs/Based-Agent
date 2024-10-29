import os
from dotenv import load_dotenv
from pathlib import Path
import time
import logging
from swarm import Swarm, Agent
import backoff
from openai import OpenAI, RateLimitError
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
env_path = BASE_DIR / '.env'
load_dotenv(env_path)

# Verify the API key is loaded
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
# Global variables
strategy_history = []

@backoff.on_exception(
    backoff.expo,
    RateLimitError,
    max_tries=8,
    max_time=60,
    on_backoff=lambda details: logger.info(f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries")
)
def get_completion(model: str, messages: list):
    """
    Get completion from OpenAI with backoff retry logic
    
    Args:
        model (str): The model to use
        messages (list): The messages to send
    
    Returns:
        dict: The API response
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response
    except RateLimitError as e:
        logger.warning(f"Rate limit hit, backing off: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in API call: {str(e)}")
        raise

async def develop_trading_strategy(market_data: str, personality: dict, model: str):
    """
    Develops a trading strategy based on market data and personality
    
    Args:
        market_data (str): JSON string of market data
        personality (dict): Trading personality configuration
        model (str): The model to use for generation
    
    Returns:
        dict: The developed strategy
    """
    try:
        messages = [
            {
                "role": "system",
                "content": f"""
                You are a trading strategy agent with the following personality:
                Name: {personality.get('name', 'Generic Trader')}
                Description: {personality.get('description', 'A balanced trading approach')}
                Risk Tolerance: {personality.get('risk_tolerance', 'medium')}
                Time Horizon: {personality.get('time_horizon', 'medium')}
                Preferred Indicators: {', '.join(personality.get('preferred_indicators', ['general']))}
                
                Analyze the market data and develop a strategy that matches your personality.
                """
            },
            {
                "role": "user",
                "content": f"Analyze this market data and develop a strategy: {market_data}"
            }
        ]
        
        response = get_completion(model, messages)
        
        strategy = {
            'name': personality.get('name', 'Generic Trader'),
            'analysis': response.choices[0].message.content,
            'timestamp': time.time(),
            'personality': personality
        }
        
        strategy_history.append(strategy)
        logger.info(f"Strategy development completed for {personality.get('name')}")
        return strategy
        
    except Exception as e:
        logger.error(f"Error in strategy development: {str(e)}", exc_info=True)
        raise

class TradingStrategyAgent(Agent):
    def __init__(self, name: str, model: str = "gpt-3.5-turbo", personality: dict = None):

        instructions = f"""
        You are a trading strategy agent that develops and executes trading strategies
        on Base. Your trading personality is:
        
        Name: {personality.get('name', 'Generic Trader')}
        Description: {personality.get('description', 'A balanced trading approach')}
        Risk Tolerance: {personality.get('risk_tolerance', 'medium')}
        Time Horizon: {personality.get('time_horizon', 'medium')}
        Preferred Indicators: {', '.join(personality.get('preferred_indicators', ['general']))}
        
        When analyzing market data:
        1. Look for clear patterns and trends
        2. Consider risk management parameters aligned with your risk tolerance
        3. Propose specific entry and exit points
        4. Document your reasoning
        5. Stay true to your trading personality and preferred approach
        """
        
        super().__init__(
            name=name,
            model=model,
            instructions=instructions,
            functions=[
                develop_trading_strategy,
                get_completion
            ],
            use_async=True
        )
        
        logger.info(f"TradingStrategyAgent initialized with model: {model} and personality: {name}")

    async def develop_strategy(self, market_data, personality, model):
        """Wrapper method to develop a trading strategy"""
        return await develop_trading_strategy(market_data, personality, model)

    def get_strategy_history(self):
        """Returns the history of developed strategies"""
        return strategy_history
