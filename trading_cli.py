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

# Load environment variables
load_dotenv()

console = Console()
client = Swarm()

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
            if not market_data_path.exists():
                console.print(f"[bold red]❌ Error: Market data file not found at {market_data_path}[/bold red]")
                return
                
            with market_data_path.open('r') as f:
                market_data_content = json.load(f)
        except json.JSONDecodeError:
            console.print("[bold red]❌ Error: Invalid JSON in market data file[/bold red]")
            return
        except Exception as e:
            console.print(f"[bold red]❌ Error reading market data: {str(e)}[/bold red]")
            return

        # Run the agent
        response = await client.run(
            agent=strategy_agent,
            messages=[{
                "role": "user", 
                "content": f"Analyze this market data and develop a strategy: {json.dumps(market_data_content, indent=2)}"
            }]
        )
        
        # Show the agent's response
        strategy_panel = Panel(
            str(response.messages[-1]["content"]),
            title="[bold green]📊 Proposed Trading Strategy",
            subtitle=f"by {strategy_name}",
            border_style="green"
        )
        console.print(strategy_panel)
        
        return response
        
    except Exception as e:
        console.print(f"[bold red]❌ Error proposing strategy: {str(e)}[/bold red]")

if __name__ == '__main__':
    cli()
