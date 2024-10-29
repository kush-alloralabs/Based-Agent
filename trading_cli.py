import asyncio
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from trading_strategy_agent import TradingStrategyAgent
from swarm import Swarm
import os
from dotenv import load_dotenv
import json
from pathlib import Path
from trading_personalities import TRADING_PERSONALITIES
from judge_agent import JudgeAgent
import backoff
from openai import OpenAI, RateLimitError

#
console = Console()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def coro(f):
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

@click.group()
def cli():
    """Trading Agent CLI - Simulate trading strategy proposals and evaluations"""
    pass

@cli.command(name='propose')
@click.option('--strategy-name', prompt='Strategy name', help='Name of the trading strategy')
@click.option('--market-data', default='./data/market_data.json', 
              help='Path to market data file', type=click.Path(exists=True))
@click.option('--model', default="gpt-3.5-turbo", help='Model to use for strategy development')
@coro
async def propose_strategy(strategy_name, market_data, model):
    """Submit a trading strategy proposal"""
    try:
        console.print(f"\n[bold blue]🤖 Strategy Agent '{strategy_name}' is analyzing the market...[/bold blue]")
        
        # Initialize agent
        strategy_agent = TradingStrategyAgent(name=strategy_name, model=model)
        
        # Read market data
        try:
            market_data_path = Path(market_data).resolve()
            with market_data_path.open('r') as f:
                market_data_content = json.load(f)
        except Exception as e:
            console.print(f"[bold red]❌ Error reading market data: {str(e)}[/bold red]")
            return

        # Run the agent with backoff handling
        try:
            strategy = await strategy_agent.develop_strategy(json.dumps(market_data_content))
            
            strategy_panel = Panel(
                str(strategy['analysis']),
                title="[bold green]📊 Proposed Trading Strategy",
                subtitle=f"by {strategy_name}",
                border_style="green"
            )
            console.print(strategy_panel)
            
            return strategy
            
        except Exception as e:
            console.print(f"[bold red]❌ Error proposing strategy: {str(e)}[/bold red]")
            
    except Exception as e:
        console.print(f"[bold red]❌ Error: {str(e)}[/bold red]")

@cli.command(name='compete')
@click.option('--market-data', default='./data/market_data.json', 
              help='Path to market data file', type=click.Path(exists=True))
@click.option('--rounds', default=3, help='Number of competition rounds')
@click.option('--model', default="gpt-3.5-turbo", help='Model to use for agents')
@coro
async def run_competition(market_data, rounds, model):
    """Run a trading strategy competition between different agent personalities"""
    try:
        console.print("\n[bold blue]🏆 Starting Trading Strategy Competition[/bold blue]")
        
        # Load market data
        market_data_path = Path(market_data).resolve()
        with market_data_path.open('r') as f:
            market_data_content = json.load(f)
        
        # Initialize agents and judge
        agents = {}
        for personality_id, personality in TRADING_PERSONALITIES.items():
            print(personality)
            agent = TradingStrategyAgent(
                name=personality['name'],
                model=model,
                personality=personality
            )
            agents[personality_id] = agent
        
        judge = JudgeAgent()
        results = []
        
        # Run competition rounds
        for round_num in range(1, rounds + 1):
            console.print(f"\n[bold cyan]📊 Round {round_num} of {rounds}[/bold cyan]")
            
            round_results = {}
            for personality_id, agent in agents.items():
                console.print(f"\n[bold yellow]🤖 {agent.name} proposing strategy...[/bold yellow]")
                
                strategy = await agent.develop_strategy(json.dumps(market_data_content), TRADING_PERSONALITIES[personality_id], model)
                evaluation = await judge.evaluate_process(strategy)
                
                round_results[personality_id] = {
                    'strategy': strategy,
                    'evaluation': evaluation
                }
                
                # Display results
                strategy_panel = Panel(
                    str(strategy['analysis']),
                    title=f"[bold green]Strategy by {agent.name}",
                    subtitle=f"Score: {evaluation['overall_score']}/100",
                    border_style="green"
                )
                console.print(strategy_panel)
            
            results.append(round_results)
            
        # Determine winner
        final_scores = {}

        # Iterate over each round's results
        for round_results in results:
            # Iterate over each personality_id in the current round
            for personality_id, data in round_results.items():
                # Get the overall score for the current personality_id
                overall_score = data['evaluation']['overall_score']
                
                # Add the score to the final_scores dictionary
                if personality_id not in final_scores:
                    final_scores[personality_id] = 0
                final_scores[personality_id] += overall_score

        # Calculate the average score for each personality_id
        for personality_id in final_scores:
            final_scores[personality_id] /= rounds
            console.print(f"Personality ID: {personality_id}, Average Score: {final_scores[personality_id]}")

        winner = max(final_scores.items(), key=lambda x: x[1])
        
        # Display final results
        console.print("\n[bold green]🏆 Competition Results[/bold green]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Agent")
        table.add_column("Average Score")
        table.add_column("Final Rank")
        
        for rank, (personality_id, score) in enumerate(
            sorted(final_scores.items(), key=lambda x: x[1], reverse=True), 1
        ):
            table.add_row(
                TRADING_PERSONALITIES[personality_id]['name'],
                f"{score:.2f}",
                f"#{rank}"
            )
        
        console.print(table)
        console.print(f"\n[bold green]🎉 Winner: {TRADING_PERSONALITIES[winner[0]]['name']}![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Competition error: {str(e)}[/bold red]")

if __name__ == '__main__':
    cli()
