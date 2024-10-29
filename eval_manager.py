from judge_agent import JudgeAgent
from trading_strategy_agent import TradingStrategyAgent

class TradingEvaluationManager:
    def __init__(self):
        self.judge = JudgeAgent()
        self.strategies = {}
        self.evaluations = {}
        
    async def evaluate_strategy_agent(self, agent: TradingStrategyAgent, market_data):
        """
        Manages the evaluation process of a trading strategy agent
        """
        try:
            # Run strategy development
            workspace = await agent.develop_strategy(market_data)
            
            # Evaluate using judge agent
            evaluation = await self.judge.evaluate_process(
                agent_workspace=workspace,
                trajectory_data=agent.trajectory
            )
            
            self.evaluations[agent.name] = evaluation
            return evaluation
            
        except Exception as e:
            return f"Evaluation failed: {str(e)}"
