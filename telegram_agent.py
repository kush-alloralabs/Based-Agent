from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import TimedOut, NetworkError, BadRequest
from trading_strategy_agent import TradingStrategyAgent
import asyncio
from typing import Dict, List, Optional
from pydantic import BaseModel
import time
from rich.console import Console
import random


console = Console()

class TelegramCommunicationAgent(TradingStrategyAgent):
    class Config:
        arbitrary_types_allowed = True
        
    def __init__(self, name: str, telegram_token: str, group_id: str = None, model: str = "gpt-4o", personality: dict = None):
        super().__init__(name=name, model=model, personality=personality)
        self._telegram_token = telegram_token
        self._telegram_bot = None
        self._group_chat_id = group_id
        self._chat_history: Dict[str, List[dict]] = {}
        self._other_agents: Dict[str, 'TelegramCommunicationAgent'] = {}
        self._app: Optional[Application] = None
        self._personality = personality or {}
        self._update_id = 0  # Add this line to track last update
        
    @property
    def telegram_bot(self):
        if not self._telegram_bot:
            self._telegram_bot = Bot(self._telegram_token)
        return self._telegram_bot
    
    @property
    def app(self) -> Optional[Application]:
        return self._app
    
    @app.setter
    def app(self, value: Application):
        self._app = value
    
    @property
    def group_chat_id(self):
        return self._group_chat_id
    
    @group_chat_id.setter
    def group_chat_id(self, value):
        self._group_chat_id = value
        
    @property
    def chat_history(self):
        return self._chat_history
        
    @property
    def other_agents(self) -> Dict[str, 'TelegramCommunicationAgent']:
        """Get registered other agents"""
        return self._other_agents
        
    @property
    def personality(self):
        return self._personality

    async def verify_group_access(self, max_retries=3):
        """Verify bot has access to the group with retries"""
        for attempt in range(max_retries):
            try:
                chat = await self.telegram_bot.get_chat(self._group_chat_id)
                console.print(f"[green]✓ Connected to group: {chat.title}[/green]")
                return True
            except (TimedOut, NetworkError) as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    console.print(f"[yellow]⚠️ Timeout on attempt {attempt + 1}, waiting {wait_time}s...[/yellow]")
                    await asyncio.sleep(wait_time)
                else:
                    console.print(f"[red]❌ Failed to verify group access after {max_retries} attempts[/red]")
                    raise
            except BadRequest as e:
                console.print(f"[red]❌ Invalid group ID: {str(e)}[/red]")
                raise
            except Exception as e:
                console.print(f"[red]❌ Unexpected error: {str(e)}[/red]")
                raise

    async def send_message_with_retry(self, text: str, max_retries=3):
        """Send a message with retry logic"""
        for attempt in range(max_retries):
            try:
                message = await self.telegram_bot.send_message(
                    chat_id=self._group_chat_id,
                    text=text,
                    parse_mode='HTML',
                    read_timeout=30,
                    write_timeout=30,
                    connect_timeout=30,
                    pool_timeout=30
                )
                return message
            except (TimedOut, NetworkError) as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    console.print(f"[yellow]⚠️ Timeout on attempt {attempt + 1}, waiting {wait_time}s...[/yellow]")
                    await asyncio.sleep(wait_time)
                else:
                    raise

    async def setup_telegram(self):
        """Initialize Telegram bot and handlers"""
        try:
            # Initialize application with custom timeouts and specific update ID
            self._app = (
                Application.builder()
                .token(self._telegram_token)
                .read_timeout(30)
                .write_timeout(30)
                .connect_timeout(30)
                .pool_timeout(30)
                .build()
            )

            # Get the current update ID to avoid conflicts
            try:
                updates = await self._app.bot.get_updates(offset=-1, timeout=1)
                if updates:
                    self._update_id = updates[-1].update_id + 1
            except Exception as e:
                console.print(f"[yellow]⚠️ Could not get latest update ID: {str(e)}[/yellow]")

            # Add handlers
            self._app.add_handler(CommandHandler("start", self.start_command))
            self._app.add_handler(CommandHandler("stats", self.stats_command))
            self._app.add_handler(CommandHandler("agents", self.list_agents_command))
            self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

            # Initialize application
            await self._app.initialize()
            
            # Start polling with specific update ID
            self._app.updater = self._app.updater or Application.get_updater(self._app)
            await self._app.updater.initialize()
            await self._app.start()
            await self._app.updater.start_polling(
                poll_interval=1.0,
                timeout=30,
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES,
                read_timeout=30,
                write_timeout=30,
                connect_timeout=30,
                pool_timeout=30,
                bootstrap_retries=5,
            )
            
            console.print(f"[green]✓ Bot polling started for {self.name} with update_id {self._update_id}[/green]")

            # Verify group access and send intro message
            if not self._group_chat_id:
                console.print("\n[yellow]Please provide a group ID using --group-id option[/yellow]")
                raise ValueError("No group ID provided")

            await self.verify_group_access()

            # Send introduction message
            intro_msg = (
                f"🤖 Trading Agent {self.name} is active!\n\n"
                f"My personality: {self.personality.get('description', 'No description')}\n"
                f"Try these commands:\n"
                f"/start - Get started\n"
                f"/stats - View statistics\n"
                f"/agents - List all agents\n\n"
                f"I'm now listening for commands! 👂"
            )

            message = await self.send_message_with_retry(intro_msg)
            
            # Start conversation loop in the background
            asyncio.create_task(self.start_conversation_loop())
            
            console.print(f"[green]✓ Agent {self.name} successfully initialized and listening for commands[/green]")

        except Exception as e:
            console.print(f"[bold red]❌ Error in setup_telegram: {str(e)}[/bold red]")
            raise

    async def cleanup(self):
        """Cleanup bot resources"""
        try:
            if self._app and self._app.updater:
                await self._app.updater.stop()
            if self._app:
                await self._app.stop()
            console.print(f"[green]✓ Cleaned up resources for {self.name}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error cleaning up {self.name}: {str(e)}[/red]")

    def register_agent(self, name: str, agent: 'TelegramCommunicationAgent'):
        """Register another agent for inter-agent communication"""
        self._other_agents[name] = agent

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        try:
            welcome_message = (
                f"👋 Hello! I am {self.name}, your trading assistant.\n\n"
                f"🤖 My personality: {self.personality.get('description', 'No description')}\n"
                f"📊 Risk tolerance: {self.personality.get('risk_tolerance', 'Moderate')}\n"
                f"⏱️ Time horizon: {self.personality.get('time_horizon', 'Medium-term')}\n\n"
                f"Available commands:\n"
                f"/start - Show this message\n"
                f"/stats - View my trading statistics\n"
                f"/agents - List all active agents"
            )
            await update.message.reply_text(welcome_message)
            console.print(f"[green]✓ Responded to /start command in group[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error in start_command: {str(e)}[/red]")
            await update.message.reply_text("Sorry, there was an error processing your command.")

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        try:
            stats_message = (
                f"📊 Trading Stats for {self.name}\n\n"
                f"Personality: {self.personality.get('description', 'No description')}\n"
                f"Risk Tolerance: {self.personality.get('risk_tolerance', 'Moderate')}\n"
                f"Time Horizon: {self.personality.get('time_horizon', 'Medium-term')}\n"
                f"Messages Processed: {len(self.chat_history.get(str(update.effective_chat.id), []))}\n"
                f"Active Since: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            await update.message.reply_text(stats_message)
            console.print(f"[green]✓ Responded to /stats command in group[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error in stats_command: {str(e)}[/red]")
            await update.message.reply_text("Sorry, there was an error processing your command.")

    async def list_agents_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /agents command"""
        try:
            agents_message = "🤖 Active Trading Agents:\n\n"
            agents_message += f"• {self.name} ({self.personality.get('description', 'No description')})\n"
            
            for name, agent in self.other_agents.items():
                agents_message += f"• {name} ({agent.personality.get('description', 'No description')})\n"
            
            await update.message.reply_text(agents_message)
            console.print(f"[green]✓ Responded to /agents command in group[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error in list_agents_command: {str(e)}[/red]")
            await update.message.reply_text("Sorry, there was an error processing your command.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Process incoming messages"""
        console.print(f"[blue]💬 Received message: {update.message.text}[/blue]")
        try:
            message_text = update.message.text
            chat_id = str(update.effective_chat.id)
            
            # Add message to chat history
            if chat_id not in self.chat_history:
                self.chat_history[chat_id] = []
            
            self.chat_history[chat_id].append({
                "role": "user",
                "content": message_text
            })
            
            # Decide whether to respond based on personality
            should_respond = await self.evaluate_response_need(message_text)
            
            if should_respond:
                response = await self.generate_response(chat_id, message_text)
                await self.propagate_message(chat_id, response)
                await update.message.reply_text(response)
                console.print(f"[green]✓ {self.name} responded to message in group[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error in handle_message: {str(e)}[/red]")

    async def evaluate_response_need(self, message: str) -> bool:
        """Evaluate whether the agent should respond based on personality"""
        evaluation_prompt = f"""
        Given your personality:
        {self.personality}
        
        Should you respond to this message: "{message}"?
        Consider your trading style, risk tolerance, and expertise.
        Return only 'true' or 'false'.
        """
        
        response = await self.get_completion(self.model, [
            {"role": "system", "content": evaluation_prompt},
            {"role": "user", "content": message}
        ])
        
        return response.choices[0].message.content.strip().lower() == 'true'

    async def generate_response(self, chat_id: str, message: str) -> str:
        """Generate a response based on personality and chat history"""
        context = self._chat_history[chat_id][-5:]  # Last 5 messages for context
        
        response = await self.get_completion(self.model, [
            {"role": "system", "content": self.instructions},
            *context,
            {"role": "user", "content": message}
        ])
        
        return response.choices[0].message.content

    async def propagate_message(self, chat_id: str, message: str):
        """Propagate message to other agents"""
        for agent_id, agent in self._other_agents.items():
            await agent.receive_message(chat_id, self.name, message)

    async def start_conversation_loop(self):
        """Start the conversation loop for random message generation"""
        while True:
            try:
                # Random delay between 10-20 seconds
                delay = random.uniform(10, 20)
                await asyncio.sleep(delay)
                
                # Get the last message from chat history
                chat_id = str(self._group_chat_id)
                last_message = None
                if chat_id in self.chat_history and self.chat_history[chat_id]:
                    last_message = self.chat_history[chat_id][-1]["content"]
                
                # Generate and send response
                if last_message:
                    response = await self.generate_conversation_response(chat_id, last_message)
                    message = await self.send_message_with_retry(response)
                    
                    # Add to chat history
                    self.chat_history[chat_id].append({
                        "role": "assistant",
                        "content": response,
                        "name": self.name
                    })
                    
                    console.print(f"[green]✓ {self.name} sent message: {response}[/green]")
                    
            except Exception as e:
                console.print(f"[red]❌ Error in conversation loop: {str(e)}[/red]")
                await asyncio.sleep(5)  # Wait before retrying

    async def generate_conversation_response(self, chat_id: str, last_message: str) -> str:
        """Generate a conversational response based on the last message"""
        context = []
        
        # Get last few messages for context
        if chat_id in self.chat_history:
            context = self.chat_history[chat_id][-5:]  # Last 5 messages
        
        prompt = f"""
        You are {self.name}, a trading agent with the following personality:
        {self.personality}
        
        Respond to this message in a conversational way, keeping your response brief (1-2 sentences):
        "{last_message}"
        
        Stay in character and discuss trading-related topics.
        """
        
        response = await self.get_completion(self.model, [
            {"role": "system", "content": prompt},
            *context,
            {"role": "user", "content": last_message}
        ])
        
        return response.choices[0].message.content
