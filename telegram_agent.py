from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import TimedOut, NetworkError, BadRequest
from trading_strategy_agent import TradingStrategyAgent, agent_wallet
import asyncio
from typing import Dict, List, Optional, Set, Any
from rich.console import Console
import random
import time
from openai import OpenAI


console = Console()

class AgentMessage:
    def __init__(self, sender: str, content: str, timestamp: float = None):
        self.sender = sender
        self.content = content
        self.timestamp = timestamp or time.time()
        
class AgentConversation:
    def __init__(self):
        self.messages: List[AgentMessage] = []
        self.participants: Set[str] = set()
        self.last_activity = time.time()
        
    def add_message(self, message: AgentMessage):
        self.messages.append(message)
        self.participants.add(message.sender)
        self.last_activity = time.time()

class TelegramCommunicationAgent(TradingStrategyAgent):
    def __init__(self, name: str, telegram_token: str, group_id: str = None, 
                 model: str = "gpt-4", personality: dict = None):
        # Initialize base class with protected attributes
        self._name = name
        self._model = model
        self._personality = personality or {}
        self._trajectory = []
        self._wallet = agent_wallet
        
        # Telegram-specific attributes
        self._telegram_token = telegram_token
        self._telegram_bot = None
        self._group_chat_id = group_id
        self._conversations = {}
        self._other_agents = {}
        self._app = None
        self._update_id = 0
        self._is_first_agent = False
        self._chat_history = {}
        self._agent_states = {}
        self.client = OpenAI()  # Initialize OpenAI client

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
            # Initialize the application with proper defaults
            builder = Application.builder()
            builder.token(self._telegram_token)
            builder.read_timeout(30)
            builder.write_timeout(30)
            builder.connect_timeout(30)
            builder.get_updates_read_timeout(30)
            builder.pool_timeout(30)
            
            self._app = builder.build()
            
            # Add message handler with high priority
            self._app.add_handler(
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND, 
                    self.handle_message,
                    block=False
                ),
                group=0  # Higher priority
            )
            
            # Add command handlers
            self._app.add_handler(CommandHandler("start", self.start_command))
            self._app.add_handler(CommandHandler("stats", self.stats_command))
            self._app.add_handler(CommandHandler("agents", self.list_agents_command))
            
            # Start the bot
            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling(
                poll_interval=1.0,  # Slightly increased interval
                timeout=30,
                drop_pending_updates=True,
                allowed_updates=[Update.MESSAGE],
                read_timeout=30,
                bootstrap_retries=5
            )
            
            console.print(f"[green]✓ Bot polling started for {self.name}[/green]")
            
            # Verify group access and start conversation loop
            if self._group_chat_id:
                await self.verify_group_access()
                # Start conversation loop in the background
                self._conversation_task = asyncio.create_task(self.start_conversation_loop())
                console.print(f"[green]✓ Started conversation loop for {self.name}[/green]")
            else:
                console.print("[yellow]Warning: No group ID provided[/yellow]")
                
        except Exception as e:
            console.print(f"[bold red]❌ Error in setup_telegram: {str(e)}[/bold red]")
            raise

    async def cleanup(self):
        """Cleanup bot resources"""
        try:
            # Cancel the conversation loop task if it exists
            if hasattr(self, '_conversation_task'):
                self._conversation_task.cancel()
                try:
                    await self._conversation_task
                except asyncio.CancelledError:
                    pass
                
            if self._app and self._app.updater:
                await self._app.updater.stop()
            if self._app:
                await self._app.stop()
            console.print(f"[green]✓ Cleaned up resources for {self.name}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error cleaning up {self.name}: {str(e)}[/red]")

    async def register_agent(self, name: str, agent: 'TelegramCommunicationAgent'):
        """Register another agent with enhanced state tracking"""
        console.print(f"[yellow]Debug: {self.name} registering agent {name}[/yellow]")
        self._other_agents[name] = agent
        self._agent_states[name] = {
            "last_interaction": time.time(),
            "conversation_count": 0,
            "response_rate": 1.0,  # Initial response rate
            "personality": agent.personality
        }
        console.print(f"[green]✓ {self.name} registered agent {name}[/green]")

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
        """Handle all incoming messages (both from Telegram and other agents)"""
        try:
            if not update.message or not update.message.text:
                return
                
            chat_id = str(update.message.chat_id)
            message_text = update.message.text
            sender_name = update.message.from_user.first_name
            
            # Update the last processed update_id
            if hasattr(update, 'update_id'):
                self._app.updater._last_update_id = update.update_id
            
            console.print(f"[yellow]Debug: {self.name} received message: {message_text} from {sender_name}[/yellow]")
            
            # Initialize chat history for new chats
            if chat_id not in self._chat_history:
                self._chat_history[chat_id] = []
                
            # Add user message to history
            self._chat_history[chat_id].append({
                "role": "user",
                "content": message_text,
                "user_id": update.message.from_user.id,
                "sender_name": sender_name
            })
            
            # Get list of potential responders (all agents except sender)
            potential_responders = [name for name in self._other_agents.keys() if name != sender_name]
            if self.name != sender_name:
                potential_responders.append(self.name)
            
            # If this bot is chosen as the responder, generate and send response
            if potential_responders and self.name == random.choice(potential_responders):
                try:
                    # Add random delay (1-3 seconds)
                    await asyncio.sleep(random.uniform(1, 3))
                    
                    # Generate response using generate_conversation_response
                    response = await self.generate_conversation_response(chat_id, message_text)
                    
                    console.print(f"[yellow]Debug: {self.name} generated response: {response}[/yellow]")
                    
                    # Add response to history
                    self._chat_history[chat_id].append({
                        "role": "assistant",
                        "content": response,
                        "name": self.name
                    })
                    
                    # Send response
                    await update.message.reply_text(response)
                    console.print(f"[green]✓ {self.name} responded to {sender_name}[/green]")
                    
                except Exception as e:
                    error_msg = f"Error processing message: {str(e)}"
                    console.print(f"[red]{error_msg}[/red]")
                    await update.message.reply_text("Sorry, I encountered an error processing your message.")

        except Exception as e:
            console.print(f"[red]Error in handle_message for {self.name}: {str(e)}[/red]")
            raise

    async def generate_conversation_response(self, chat_id: str, last_message: str) -> str:
        """Generate a conversational response based on the last message"""
        context = []
        
        # Get last few messages for context
        if chat_id in self._conversations:
            # Get the last 3 messages for better context
            context = [
                {"role": "user" if msg.sender != self.name else "assistant", 
                 "content": msg.content}
                for msg in list(self._conversations[chat_id].messages)[-3:]
            ]
        
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

    async def propagate_message(self, chat_id: str, message: str):
        """Enhanced message propagation with selective forwarding"""
        message_obj = AgentMessage(sender=self.name, content=message)
        
        # Ensure conversation exists
        if chat_id not in self._conversations:
            self._conversations[chat_id] = AgentConversation()
        
        # Add message to conversation
        self._conversations[chat_id].add_message(message_obj)
        
        # Determine which agents should receive the message
        relevant_agents = await self._filter_relevant_agents(message)
        
        # Propagate to relevant agents
        for agent_name, agent in self._other_agents.items():
            console.print(f"[yellow]Debug: {self.name} propagating message to {agent_name}[/yellow]")
            if agent_name in relevant_agents:
                await agent.receive_message(chat_id, self.name, message)
                self._update_agent_state(agent_name, "message_sent")

    async def receive_message(self, chat_id: str, sender: str, content: str):
        """Handle received messages with guaranteed response"""
        try:
            # Never process our own messages
            if sender == self.name:
                console.print(f"[yellow]Debug: {self.name} skipping own message[/yellow]")
                return
                
            # Track this message as processed
            message_hash = hash(f"{sender}:{content}")
            if not hasattr(self, '_processed_messages'):
                self._processed_messages = set()
            
            # Skip if we've already processed this message
            if message_hash in self._processed_messages:
                console.print(f"[yellow]Debug: {self.name} already processed message from {sender}: {content[:30]}...[/yellow]")
                return
            
            console.print(f"[yellow]Debug: {self.name} processing new message from {sender}: {content[:30]}...[/yellow]")
            self._processed_messages.add(message_hash)
            
            # Add message to conversation history
            message_obj = AgentMessage(sender=sender, content=content)
            if chat_id not in self._conversations:
                self._conversations[chat_id] = AgentConversation()
            self._conversations[chat_id].add_message(message_obj)
            
            # Get ALL available responders except sender
            available_responders = [
                name for name in list(self._other_agents.keys()) + [self.name]
                if name != sender
            ]
            
            # For initial messages (from conversation_loop), ensure someone responds
            initial_keywords = ["BTC", "ETH", "USDT", "BNB"]
            is_initial_message = any(keyword in content for keyword in initial_keywords)
            
            # Modified selection logic to ensure continued conversation
            if is_initial_message:
                selected_responder = min(available_responders)
            else:
                # Use combination of message and previous sender for selection
                combined_hash = hash(f"{sender}:{content}")
                selected_responder = available_responders[abs(combined_hash) % len(available_responders)]
                console.print(f"[yellow]Debug: Selected {selected_responder} to respond to {sender}'s message[/yellow]")
            
            # If this bot is the selected responder
            if self.name == selected_responder:
                console.print(f"[yellow]Debug: {self.name} was selected to respond to {sender}[/yellow]")
                await asyncio.sleep(random.uniform(1, 3))
                response = await self.generate_conversation_response(chat_id, content)
                
                # Send directly via bot
                await self._app.bot.send_message(chat_id=self._group_chat_id, text=response)
                console.print(f"[green]✓ {self.name} responded to {sender}[/green]")

                # Propagate response to other agents
                for agent_name, agent in self._other_agents.items():
                    console.print(f"[yellow]Debug: {self.name} propagating message to {agent_name}[/yellow]")
                    await agent.receive_message(str(self._group_chat_id), self.name, response)
                
        except Exception as e:
            console.print(f"[red]Error in receive_message for {self.name}: {str(e)}[/red]")
            raise

    async def _filter_relevant_agents(self, message: str) -> Set[str]:
        """Determine which agents should receive a message based on content and context"""
        relevant_agents = set()
        
        for agent_name, agent in self._other_agents.items():
            # Check agent's expertise and interests
            relevant_agents.add(agent_name)
                
        return relevant_agents

    def _update_agent_state(self, agent_name: str, action: str):
        """Update agent interaction states"""
        if agent_name in self._agent_states:
            state = self._agent_states[agent_name]
            state["last_interaction"] = time.time()
            
            if action == "message_sent":
                state["conversation_count"] += 1
            elif action == "response_sent":
                state["response_rate"] = (
                    state["response_rate"] * 0.9 + 0.1
                )  # Exponential moving average

    def _get_agent_relationships(self) -> Dict[str, float]:
        """Calculate relationship strengths between agents"""
        relationships = {}
        for agent_name, state in self._agent_states.items():
            interaction_score = min(state["conversation_count"] / 100, 1.0)
            response_score = state["response_rate"]
            relationships[agent_name] = (interaction_score + response_score) / 2
        return relationships

    def _generate_system_prompt(self, context: dict) -> str:
        """Generate enhanced system prompt with agent context"""
        return f"""You are {self.name}, a trading agent with the following personality:
{self.personality}

Current conversation context:
- Interacting with: {context['sender']}
- Relationship strength: {context['agent_relationships'].get(context['sender'], 0):.2f}
- Other participants: {', '.join(context['conversation_participants'])}

Maintain your unique personality while considering:
1. Your relationship with the sender
2. The conversation history
3. The presence of other agents

Respond naturally while staying true to your trading expertise and character."""

    async def start_conversation_loop(self):
        """Start the conversation loop with intelligent conversation starters"""
        try:
            # Initial delay to allow all agents to initialize
            await asyncio.sleep(30)
            
            # Only check once if this bot should start the conversation
            should_start = (
                self._group_chat_id and  # Ensure we have a group chat ID
                self._other_agents and   # Ensure other agents are registered
                self.name == min(list(self._other_agents.keys()) + [self.name])  # First agent alphabetically
            )
            
            console.print(f"[yellow]Debug: {self.name} should_start = {should_start}[/yellow]")
            
            if should_start:
                console.print(f"[green]{self.name} initiating conversation...[/green]")
                
                starter_templates = [
                    "Hey traders! I've been analyzing {asset} and noticed {observation}. What do you think?",
                    "Interesting market movements today. Has anyone else spotted the {pattern} in {asset}?",
                    "I'm seeing some unusual activity in {asset}. Should we discuss potential strategies?",
                    "What's everyone's take on {asset} current price action? I'm seeing {pattern}.",
                ]
                
                # Generate initial message
                template = random.choice(starter_templates)
                asset = random.choice(["BTC", "ETH", "USDT", "BNB"])
                pattern = random.choice([
                    "a bullish divergence",
                    "an emerging triangle pattern",
                    "increasing volume",
                    "unusual volatility",
                    "a potential breakout setup"
                ])
                
                message = template.format(asset=asset, pattern=pattern, observation=pattern)
                
                # Send message to Telegram
                await self._app.bot.send_message(chat_id=self._group_chat_id, text=message)
                console.print(f"[yellow]Debug: {self.name} sent initial message: {message}[/yellow]")
                
                # Propagate message to other agents
                for agent_name, agent in self._other_agents.items():
                    console.print(f"[yellow]Debug: {self.name} propagating message to {agent_name}[/yellow]")
                    await agent.receive_message(str(self._group_chat_id), self.name, message)
                
            # After starting (or not starting) the conversation, just keep the loop alive
            while True:
                await asyncio.sleep(60)  # Sleep to prevent CPU usage
                
        except Exception as e:
            console.print(f"[red]Error in conversation loop for {self.name}: {str(e)}[/red]")

    async def get_completion(self, model: str, messages: list) -> Any:
        """Get completion from OpenAI API"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=model,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                temperature=0.7,
                max_tokens=150
            )
            return response
        except Exception as e:
            console.print(f"[red]Error getting completion: {str(e)}[/red]")
            raise

    async def send_message(self, message: str):
        """Send a message to the group chat and propagate to other agents"""
        try:
            # Send to Telegram
            await self.send_message_with_retry(message)
            console.print(f"[green]✓ {self.name} sent message: {message}[/green]")
            
            # Propagate to other agents
            for agent_name, agent in self._other_agents.items():
                console.print(f"[yellow]Debug: {self.name} propagating message to {agent_name}[/yellow]")
                await agent.receive_message(str(self._group_chat_id), self.name, message)
                
        except Exception as e:
            console.print(f"[red]Error sending message from {self.name}: {str(e)}[/red]")
            raise
