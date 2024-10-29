import logging
from swarm import Swarm, Agent
from openai import OpenAI, RateLimitError
import backoff
import time
import os
from dotenv import load_dotenv
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variables
evaluation_memory = []
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
env_path = BASE_DIR / '.env'
load_dotenv(env_path)

# Verify the API key is loaded
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

def evaluate_strategy(strategy_data):
    """
    Evaluates a single trading strategy
    
    Args:
        strategy_data (dict): The strategy to evaluate
        
    Returns:
        dict: Evaluation results
    """
    try:
        evaluation = {
            'strategy_name': strategy_data.get('name', 'Unnamed Strategy'),
            'process_quality': 80,
            'risk_management': 75,
            'market_adaptation': 70,
            'overall_score': 75,
            'evaluation_text': "Test evaluation",
            'timestamp': time.time()
        }
        evaluation_memory.append(evaluation)
        return evaluation
    except Exception as e:
        logger.error(f"Strategy evaluation failed: {str(e)}")
        return {
            'strategy_name': strategy_data.get('name', 'Unnamed Strategy'),
            'evaluation_text': f"Evaluation failed: {str(e)}",
            'overall_score': 0,
            'timestamp': time.time()
        }

class JudgeAgent(Agent):
    def __init__(self):
        logger.debug("Starting JudgeAgent initialization")
        
        # Define judge instructions
        instructions = """
        You are an expert trading strategy evaluator that analyzes both the final outcomes 
        and intermediate steps of trading strategies. You evaluate based on:
        1. Strategy robustness (0-25 points)
        2. Risk management (0-25 points)
        3. Process quality (0-20 points)
        4. Implementation correctness (0-15 points)
        5. Market adaptability (0-15 points)

        For each strategy, provide a detailed evaluation with specific scores in each category
        and clear justification for the scores. Sum these for an overall score out of 100.
        """
        
        # Initialize the base agent
        super().__init__(
            name="Trading Judge",
            model="gpt-4",
            instructions=instructions,
            functions=[
                evaluate_strategy,
            ]
        )
        logger.debug("Completed JudgeAgent initialization")

    async def evaluate_process(self, strategy_data, trajectory_data=None):
        """
        Evaluates a trading strategy and returns structured feedback
        """
        logger.debug("Starting strategy evaluation process")
        try:
            evaluation = evaluate_strategy(strategy_data)
            return evaluation
        except Exception as e:
            logger.error(f"Evaluation process failed: {str(e)}")
            return {
                'strategy_name': strategy_data.get('name', 'Unnamed Strategy'),
                'evaluation_text': f"Evaluation failed: {str(e)}",
                'overall_score': 0,
                'timestamp': time.time()
            }

    def get_evaluation_history(self):
        """
        Returns the evaluation history
        """
        return evaluation_memory
