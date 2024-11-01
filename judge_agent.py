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
        # Example logic to dynamically calculate scores
        process_quality = calculate_process_quality(strategy_data)
        risk_management = calculate_risk_management(strategy_data)
        market_adaptation = calculate_market_adaptation(strategy_data)
        
        # Calculate overall score as a weighted sum of individual scores
        overall_score = (
            process_quality * 0.25 +
            risk_management * 0.25 +
            market_adaptation * 0.20 +
            calculate_implementation_correctness(strategy_data) * 0.15 +
            calculate_strategy_robustness(strategy_data) * 0.15
        )
        
        evaluation = {
            'strategy_name': strategy_data.get('name', 'Unnamed Strategy'),
            'process_quality': process_quality,
            'risk_management': risk_management,
            'market_adaptation': market_adaptation,
            'overall_score': overall_score,
            'evaluation_text': generate_evaluation_text(strategy_data),
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

def get_strategy_history(self):
    """Returns the history of developed strategies"""
    return strategy_history

def calculate_process_quality(strategy_data):
    # Implement logic to evaluate process quality
    return 80  # Example static value, replace with dynamic logic

def calculate_risk_management(strategy_data):
    # Implement logic to evaluate risk management
    return 75  # Example static value, replace with dynamic logic

def calculate_market_adaptation(strategy_data):
    # Implement logic to evaluate market adaptation
    return 70  # Example static value, replace with dynamic logic

def calculate_implementation_correctness(strategy_data):
    # Implement logic to evaluate implementation correctness
    return 65  # Example static value, replace with dynamic logic

def calculate_strategy_robustness(strategy_data):
    # Implement logic to evaluate strategy robustness
    return 60  # Example static value, replace with dynamic logic

def generate_evaluation_text(strategy_data):
    # Generate a detailed evaluation text based on the strategy data
    return "Detailed evaluation based on strategy analysis."

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
                get_strategy_history,
                calculate_process_quality,
                calculate_risk_management,
                calculate_market_adaptation,
                calculate_implementation_correctness,
                calculate_strategy_robustness,
                generate_evaluation_text
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
