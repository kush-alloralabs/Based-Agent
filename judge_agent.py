import time
from agents import Agent

class JudgeAgent(Agent):
    def __init__(self):
        super().__init__(name="Trading Judge", instructions="""
        You are an expert trading strategy evaluator that analyzes both the final outcomes 
        and intermediate steps of trading strategies. You evaluate based on:
        1. Strategy robustness
        2. Risk management
        3. Process quality
        4. Implementation correctness
        5. Market adaptability
        """)
        self.evaluation_memory = []
        
    async def evaluate_process(self, agent_workspace, trajectory_data):
        """
        Evaluates the entire trading process, not just outcomes
        """
        try:
            # Graph module - analyze workspace structure
            workspace_graph = self.analyze_workspace_structure(agent_workspace)
            
            # Read module - parse trading results and logs
            trading_results = self.parse_trading_data(agent_workspace)
            
            # Search module - find relevant code implementations
            strategy_impl = self.search_strategy_code(agent_workspace)
            
            evaluation = {
                'process_quality': self.evaluate_process_quality(trajectory_data),
                'code_quality': self.evaluate_code_quality(strategy_impl),
                'risk_management': self.evaluate_risk_metrics(trading_results),
                'market_adaptation': self.evaluate_market_adaptation(trading_results),
                'overall_score': 0  # Will be calculated based on above
            }
            
            self.evaluation_memory.append(evaluation)
            return evaluation
            
        except Exception as e:
            return f"Evaluation failed: {str(e)}"
