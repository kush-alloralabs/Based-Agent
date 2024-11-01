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
from judge_agent import JudgeAgent
import backoff
from openai import OpenAI, RateLimitError
from trading_strategy_agent import get_completion
from trading_personalities import TRADING_PERSONALITIES
from telegram_agent import TelegramCommunicationAgent
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
@click.option('--model', default="gpt-4o", help='Model to use for strategy development')
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
@click.option('--model', default="gpt-4o", help='Model to use for agents')
@click.option('--context-file', default='./data/context.rtf', 
              help='Path to context RTF file', type=click.Path(exists=True))
@coro
async def run_competition(market_data, rounds, model, context_file):
    """Run a trading strategy competition between different agent personalities"""
    try:
        console.print("\n[bold blue]🏆 Starting Trading Strategy Competition[/bold blue]")
        
        # Load market data
        market_data_path = Path(market_data).resolve()
        with market_data_path.open('r') as f:
            market_data_content = json.load(f)
            
        # Load context from RTF file
        context_path = Path(context_file).resolve()
        with context_path.open('r') as f:
            context_content = f.read()
            
        # Parse RTF content to extract relevant sections
        # You might need to adjust this based on your RTF structure
        system_context = ""
        if '<truth-terminal-openpipe:digital-twin#SYSTEM>' in context_content:
            system_context = context_content.split('<truth-terminal-openpipe:digital-twin#SYSTEM>')[1]
            system_context = system_context.split('<')[0].strip()
            
        # Initialize agents and judge
        agents = {}
        for personality_id, personality in TRADING_PERSONALITIES.items():
            # Combine personality with system context
            enhanced_personality = personality.copy()
            enhanced_personality['context'] = system_context
            
            agent = TradingStrategyAgent(
                name=personality['name'],
                model=model,
                personality=enhanced_personality
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
                
                strategy = await agent.develop_strategy(
                    json.dumps(market_data_content), 
                    TRADING_PERSONALITIES[personality_id], 
                    model
                )
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
            
        # Calculate and display final scores
        final_scores = {}
        for personality_id in TRADING_PERSONALITIES:
            scores = [round_result[personality_id]['evaluation']['overall_score'] 
                     for round_result in results]
            final_scores[personality_id] = sum(scores) / len(scores)
            
            console.print(f"Personality ID: {personality_id}, Average Score: {final_scores[personality_id]}")
            
        # Display final rankings
        console.print("\n[bold green]🏆 Competition Results[/bold green]")
        rankings = Table(show_header=True, header_style="bold magenta")
        rankings.add_column("Agent", style="dim")
        rankings.add_column("Average Score")
        rankings.add_column("Final Rank")
        
        sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (personality_id, score) in enumerate(sorted_scores, 1):
            rankings.add_row(
                TRADING_PERSONALITIES[personality_id]['name'],
                f"{score:.2f}",
                f"#{rank}"
            )
            
        console.print(rankings)
        console.print(f"\n[bold green]🎉 Winner: {TRADING_PERSONALITIES[sorted_scores[0][0]]['name']}![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {str(e)}[/bold red]")

@cli.command(name='generate-personalities')
@click.option('--num-personalities', default=5, help='Number of trading personalities to generate')
@click.option('--model', default="gpt-4o", help='Model to use for generation')
@click.option('--context-file', default='./data/context.rtf', 
              help='Path to context RTF file', type=click.Path(exists=True))
@coro
async def generate_personalities(num_personalities, model, context_file):
    """Generate trading personalities using LLM"""
    try:
        console.print("\n[bold blue]🤖 Generating Trading Personalities...[/bold blue]")
        
        # Load context from RTF file
        context_path = Path(context_file).resolve()
        with context_path.open('r') as f:
            context_content = f.read()
            
        # Parse RTF content to extract relevant sections
        system_context = ""
        if '<truth-terminal-openpipe:digital-twin#SYSTEM>' in context_content:
            system_context = context_content.split('<truth-terminal-openpipe:digital-twin#SYSTEM>')[1]
            system_context = system_context.split('<')[0].strip()
        
        messages = [{
            "role": "system",
            "content": f"""You are an expert in trading strategies and personality types.
            Using the following context as background knowledge:
            
            {system_context}
            
            Generate unique trading personalities as a Python dictionary. Example format:
            {{
                "Momentum_Trader": {{
                    "name": "Momentum Trader",
                    "description": "Focuses on assets showing strong price momentum",
                    "risk_tolerance": "high",
                    "time_horizon": "short",
                    "preferred_indicators": ["rsi", "macd", "volume"]
                }},
                "Value_Investor": {{
                    "name": "Value Investor",
                    "description": "Seeks undervalued assets for long-term growth",
                    "risk_tolerance": "low",
                    "time_horizon": "long",
                    "preferred_indicators": ["pe_ratio", "book_value", "cash_flow"]
                }}
            }}

            Requirements:
            1. Use underscores in personality_ids (e.g., "Day_Trader")
            2. Risk tolerance must be one of: "low", "medium", "medium-high", "high"
            3. Time horizon must be one of: "short", "medium", "medium-long", "long"
            4. Include 2-4 relevant technical indicators per personality
            5. Ensure all strings are properly quoted
            6. Use consistent indentation
            7. Generate personalities that would work well in the context provided
            8. Return ONLY the dictionary, no variable assignments or additional text
            """
        }, {
            "role": "user",
            "content": f"Generate {num_personalities} unique trading personalities that align with the provided context. Return only the Python dictionary, no variable assignments or additional text."
        }]
        
        response = get_completion(model, messages)
        personalities_str = response.choices[0].message.content.strip()
        
        # Clean up the response
        personalities_str = personalities_str.replace("```python", "").replace("```", "").strip()
        
        # Remove any variable assignments
        if "=" in personalities_str:
            personalities_str = personalities_str.split("=", 1)[1].strip()
        
        # Try to evaluate the string as a Python dictionary
        try:
            personalities = eval(personalities_str)
        except SyntaxError as e:
            console.print("[bold red]Failed to parse response as Python dictionary[/bold red]")
            console.print(f"Response was:\n{personalities_str}")
            raise
            
        # Validate personalities
        for personality_id, personality in personalities.items():
            validate_personality(personality)

        # Save to trading_personalities.py
        file_content = f"""# Auto-generated trading personalities
TRADING_PERSONALITIES = {personalities_str}
"""
        
        personalities_path = Path(__file__).parent / 'trading_personalities.py'
        with personalities_path.open('w') as f:
            f.write(file_content)
            
        # Display generated personalities
        console.print("\n[bold green]✨ Generated Trading Personalities:[/bold green]")
        for personality_id, personality in personalities.items():
            panel = Panel(
                f"""Name: {personality['name']}
Description: {personality['description']}
Risk Tolerance: {personality['risk_tolerance']}
Time Horizon: {personality['time_horizon']}
Preferred Indicators: {', '.join(personality['preferred_indicators'])}""",
                title=f"[bold cyan]{personality_id}[/bold cyan]",
                border_style="cyan"
            )
            console.print(panel)
            
        console.print("\n[bold green]✅ Trading personalities have been generated and saved![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error generating personalities: {str(e)}[/bold red]")
        raise

def validate_personality(personality):
    """Validate a generated trading personality"""
    required_fields = ['name', 'description', 'risk_tolerance', 'time_horizon', 'preferred_indicators']
    valid_risk_levels = ['low', 'medium', 'medium-high', 'high']
    valid_time_horizons = ['short', 'medium', 'medium-long', 'long']
    
    # Check required fields
    for field in required_fields:
        if field not in personality:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate risk tolerance
    if personality['risk_tolerance'] not in valid_risk_levels:
        raise ValueError(f"Invalid risk tolerance: {personality['risk_tolerance']}")
    
    # Validate time horizon
    if personality['time_horizon'] not in valid_time_horizons:
        raise ValueError(f"Invalid time horizon: {personality['time_horizon']}")
    
    # Validate preferred indicators
    if not isinstance(personality['preferred_indicators'], list):
        raise ValueError("preferred_indicators must be a list")
    if len(personality['preferred_indicators']) < 1:
        raise ValueError("Must have at least one preferred indicator")

@cli.command(name='start-telegram')
@click.option('--tokens', required=True, help='Comma-separated list of Telegram Bot Tokens')
@click.option('--group-id', required=True, help='Telegram group chat ID (required)')
@click.option('--timeout', default=120, help='Timeout in seconds for operations')
@coro
async def start_telegram_agents(tokens: str, group_id: str, timeout: int = 120):
    """Start trading agents with Telegram integration"""
    agents = {}
    try:
        console.print("\n[bold blue]🤖 Starting Trading Agents on Telegram...[/bold blue]")
        
        # Parse tokens
        bot_tokens = [token.strip() for token in tokens.split(',')]
        
        # Validate we have enough tokens for personalities
        if len(bot_tokens) < len(TRADING_PERSONALITIES):
            console.print(f"[yellow]Warning: Only {len(bot_tokens)} tokens provided for {len(TRADING_PERSONALITIES)} personalities[/yellow]")
        
        # Create agents with their own bot tokens
        for token, personality in zip(bot_tokens, TRADING_PERSONALITIES.values()):
            agent = TelegramCommunicationAgent(
                name=personality['name'],
                telegram_token=token,
                group_id=group_id,
                personality=personality
            )
            agents[personality['name']] = agent
            await agent.setup_telegram()
            await asyncio.sleep(2)  # Delay between agent initialization
            console.print(f"[green]✓ Created agent: {personality['name']}[/green]")

        console.print("\n[bold green]🎉 All agents are active and listening![/bold green]")
        console.print("\nAvailable commands:")
        console.print("  /start  - Get started")
        console.print("  /stats  - View agent statistics")
        console.print("  /agents - List active agents")
        console.print("\nPress Ctrl+C to stop the agents")
        
        # Keep the application running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down agents...[/yellow]")
            for agent in agents.values():
                await agent.cleanup()
            console.print("[green]✓ Agents shutdown complete[/green]")
            
    except Exception as e:
        console.print(f"[bold red]❌ Error starting Telegram agents: {str(e)}[/bold red]")
        # Attempt to clean up on error
        for agent in agents.values():
            await agent.cleanup()
        raise

if __name__ == '__main__':
    cli()
