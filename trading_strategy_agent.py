import os
from dotenv import load_dotenv
from pathlib import Path
import time
import logging
from swarm import Swarm, Agent
import backoff
from openai import OpenAI, RateLimitError
from pathlib import Path
import json
from swarm import Agent
from cdp import *
from typing import List, Dict, Any, Optional
import os
from openai import OpenAI
from decimal import Decimal
from typing import Union
from web3 import Web3
from web3.exceptions import ContractLogicError
from cdp.errors import ApiError, UnsupportedAssetError
from pydantic import BaseModel, Field
from cdp import Wallet

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

# Initialize CDP wallet only if needed
try:
    agent_wallet = Wallet()
    faucet = agent_wallet.faucet()
    logger.info(f"CDP wallet initialized: {agent_wallet.default_address.address_id}")
except Exception as e:
    logger.warning(f"CDP wallet initialization failed: {e}. Continuing without CDP functionality.")
    agent_wallet = None
    faucet = None

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
        # Load contextual information
        context = load_context(personality.get('name').replace(' ', '_'))
        
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
                
                Additional Context:
                {context}
                
                Use both your personality traits and the provided contextual information to:
                1. Analyze market conditions
                2. Develop strategies aligned with your personality
                3. Consider historical context and past decisions
                4. Make recommendations based on the complete picture
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

def load_context(personality_id: str) -> str:
    """Load contextual information for a trading personality from text files"""
    context_path = Path(__file__).parent / 'contexts' / f'{personality_id.lower()}_context.txt'
    if context_path.exists():
        with context_path.open('r') as f:
            return f.read().strip()
    return ""

def create_token(name, symbol, initial_supply):
    """
    Create a new ERC-20 token.
    
    Args:
        name (str): The name of the token
        symbol (str): The symbol of the token
        initial_supply (int): The initial supply of tokens
    
    Returns:
        str: A message confirming the token creation with details
    """
    deployed_contract = agent_wallet.deploy_token(name, symbol, initial_supply)
    deployed_contract.wait()
    return f"Token {name} ({symbol}) created with initial supply of {initial_supply} and contract address {deployed_contract.contract_address}"

# Function to transfer assets
def transfer_asset(amount, asset_id, destination_address):
    """
    Transfer an asset to a specific address.
    
    Args:
        amount (Union[int, float, Decimal]): Amount to transfer
        asset_id (str): Asset identifier ("eth", "usdc") or contract address of an ERC-20 token
        destination_address (str): Recipient's address
    
    Returns:
        str: A message confirming the transfer or describing an error
    """
    try:
        # Check if we're on Base Mainnet and the asset is USDC for gasless transfer
        is_mainnet = agent_wallet.network_id == "base-mainnet"
        is_usdc = asset_id.lower() == "usdc"
        gasless = is_mainnet and is_usdc

        # For ETH and USDC, we can transfer directly without checking balance
        if asset_id.lower() in ["eth", "usdc"]:
            transfer = agent_wallet.transfer(amount, asset_id, destination_address, gasless=gasless)
            transfer.wait()
            gasless_msg = " (gasless)" if gasless else ""
            return f"Transferred {amount} {asset_id}{gasless_msg} to {destination_address}"
            
        # For other assets, check balance first
        try:
            balance = agent_wallet.balance(asset_id)
        except UnsupportedAssetError:
            return f"Error: The asset {asset_id} is not supported on this network. It may have been recently deployed. Please try again in about 30 minutes."

        if balance < amount:
            return f"Insufficient balance. You have {balance} {asset_id}, but tried to transfer {amount}."

        transfer = agent_wallet.transfer(amount, asset_id, destination_address)
        transfer.wait()
        return f"Transferred {amount} {asset_id} to {destination_address}"
    except Exception as e:
        return f"Error transferring asset: {str(e)}. If this is a custom token, it may have been recently deployed. Please try again in about 30 minutes, as it needs to be indexed by CDP first."

# Function to get the balance of a specific asset
def get_balance(asset_id):
    """
    Get the balance of a specific asset in the agent's wallet.
    
    Args:
        asset_id (str): Asset identifier ("eth", "usdc") or contract address of an ERC-20 token
    
    Returns:
        str: A message showing the current balance of the specified asset
    """
    try:
        balance = agent_wallet.balance(asset_id)
        return f"Current balance of {asset_id}: {balance}"
    except Exception as e:
        return f"Error fetching balance for {asset_id}: {str(e)}"

# Function to request ETH from the faucet (testnet only)
def request_eth_from_faucet():
    """
    Request ETH from the Base Sepolia testnet faucet.
    
    Returns:
        str: Status message about the faucet request
    """
    if agent_wallet.network_id == "base-mainnet":
        return "Error: The faucet is only available on Base Sepolia testnet."
    
    faucet_tx = agent_wallet.faucet()
    return f"Requested ETH from faucet. Transaction: {faucet_tx}"

# Function to generate art using DALL-E (requires separate OpenAI API key)
def generate_art(prompt):
    """
    Generate art using DALL-E based on a text prompt.
    
    Args:
        prompt (str): Text description of the desired artwork
    
    Returns:
        str: Status message about the art generation, including the image URL if successful
    """
    try:
        client = OpenAI()
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        image_url = response.data[0].url
        return f"Generated artwork available at: {image_url}"
        
    except Exception as e:
        return f"Error generating artwork: {str(e)}"

# Function to deploy an ERC-721 NFT contract
def deploy_nft(name, symbol, base_uri):
    """
    Deploy an ERC-721 NFT contract.
    
    Args:
        name (str): Name of the NFT collection
        symbol (str): Symbol of the NFT collection
        base_uri (str): Base URI for token metadata
    
    Returns:
        str: Status message about the NFT deployment, including the contract address
    """
    try:
        deployed_nft = agent_wallet.deploy_nft(name, symbol, base_uri)
        deployed_nft.wait()
        contract_address = deployed_nft.contract_address
        
        return f"Successfully deployed NFT contract '{name}' ({symbol}) at address {contract_address} with base URI: {base_uri}"
        
    except Exception as e:
        return f"Error deploying NFT contract: {str(e)}"

# Function to mint an NFT
def mint_nft(contract_address, mint_to):
    """
    Mint an NFT to a specified address.
    
    Args:
        contract_address (str): Address of the NFT contract
        mint_to (str): Address to mint NFT to
    
    Returns:
        str: Status message about the NFT minting
    """
    try:
        mint_args = {
            "to": mint_to,
            "quantity": "1"
        }
        
        mint_invocation = agent_wallet.invoke_contract(
            contract_address=contract_address,
            method="mint", 
            args=mint_args
        )
        mint_invocation.wait()
        
        return f"Successfully minted NFT to {mint_to}"
        
    except Exception as e:
        return f"Error minting NFT: {str(e)}"

# Function to swap assets (only works on Base Mainnet)
def swap_assets(amount: Union[int, float, Decimal], from_asset_id: str, to_asset_id: str):
    """
    Swap one asset for another using the trade function.
    This function only works on Base Mainnet.

    Args:
        amount (Union[int, float, Decimal]): Amount of the source asset to swap
        from_asset_id (str): Source asset identifier
        to_asset_id (str): Destination asset identifier

    Returns:
        str: Status message about the swap
    """
    if agent_wallet.network_id != "base-mainnet":
        return "Error: Asset swaps are only available on Base Mainnet. Current network is not Base Mainnet."

    try:
        trade = agent_wallet.trade(amount, from_asset_id, to_asset_id)
        trade.wait()
        return f"Successfully swapped {amount} {from_asset_id} for {to_asset_id}"
    except Exception as e:
        return f"Error swapping assets: {str(e)}"

# Contract addresses for Basenames
BASENAMES_REGISTRAR_CONTROLLER_ADDRESS_MAINNET = "0x4cCb0BB02FCABA27e82a56646E81d8c5bC4119a5"
BASENAMES_REGISTRAR_CONTROLLER_ADDRESS_TESTNET = "0x49aE3cC2e3AA768B1e5654f5D3C6002144A59581"
L2_RESOLVER_ADDRESS_MAINNET = "0xC6d566A56A1aFf6508b41f6c90ff131615583BCD"
L2_RESOLVER_ADDRESS_TESTNET = "0x6533C94869D28fAA8dF77cc63f9e2b2D6Cf77eBA"

# Function to create registration arguments for Basenames
def create_register_contract_method_args(base_name: str, address_id: str, is_mainnet: bool) -> dict:
    """
    Create registration arguments for Basenames.
    
    Args:
        base_name (str): The Basename (e.g., "example.base.eth" or "example.basetest.eth")
        address_id (str): The Ethereum address
        is_mainnet (bool): True if on mainnet, False if on testnet
    
    Returns:
        dict: Formatted arguments for the register contract method
    """
    try:
        w3 = Web3()
        resolver_contract = w3.eth.contract(abi=l2_resolver_abi)
        name_hash = w3.ens.namehash(base_name)
        
        address_data = resolver_contract.encode_abi(
            "setAddr",
            args=[name_hash, address_id]
        )
        
        name_data = resolver_contract.encode_abi(
            "setName",
            args=[name_hash, base_name]
        )
        
        register_args = {
            "request": [
                base_name.replace(".base.eth" if is_mainnet else ".basetest.eth", ""),
                address_id,
                "31557600",  # 1 year in seconds
                L2_RESOLVER_ADDRESS_MAINNET if is_mainnet else L2_RESOLVER_ADDRESS_TESTNET,
                [address_data, name_data],
                True
            ]
        }
        
        return register_args
    except Exception as e:
        raise ValueError(f"Error creating registration arguments for {base_name}: {str(e)}")

# Function to register a basename
def register_basename(basename: str, amount: float = 0.002):
    """
    Register a basename for the agent's wallet.
    
    Args:
        basename (str): The basename to register (e.g. "myname.base.eth" or "myname.basetest.eth")
        amount (float): Amount of ETH to pay for registration (default 0.002)
    
    Returns:
        str: Status message about the basename registration
    """
    try:
        address_id = agent_wallet.default_address.address_id
        is_mainnet = agent_wallet.network_id == "base-mainnet"

        suffix = ".base.eth" if is_mainnet else ".basetest.eth"
        if not basename.endswith(suffix):
            basename += suffix

        register_args = create_register_contract_method_args(basename, address_id, is_mainnet)

        contract_address = (
            BASENAMES_REGISTRAR_CONTROLLER_ADDRESS_MAINNET if is_mainnet
            else BASENAMES_REGISTRAR_CONTROLLER_ADDRESS_TESTNET
        )

        invocation = agent_wallet.invoke_contract(
            contract_address=contract_address,
            method="register", 
            args=register_args,
            abi=registrar_abi,
            amount=amount,
            asset_id="eth",
        )
        invocation.wait()
        return f"Successfully registered basename {basename} for address {address_id}"
    except ContractLogicError as e:
        return f"Error registering basename: {str(e)}"
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Unexpected error registering basename: {str(e)}"


class TradingStrategyAgent:
    def __init__(self, name: str, model: str = "gpt-4", personality: dict = None):
        self._name = name
        self._model = model
        self._personality = personality or {}
        self._trajectory = []
        self._wallet = agent_wallet  # Now agent_wallet is defined

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def personality(self) -> dict:
        return self._personality

    @property
    def trajectory(self) -> list:
        return self._trajectory

    @property
    def wallet(self):
        return self._wallet

    async def develop_strategy(self, market_data, personality, model):
        """Wrapper method to develop a trading strategy"""
        return await develop_trading_strategy(market_data, personality, model)


l2_resolver_abi = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "node", "type": "bytes32"},
            {"internalType": "address", "name": "a", "type": "address"}
        ],
        "name": "setAddr",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "node", "type": "bytes32"},
            {"internalType": "string", "name": "newName", "type": "string"}
        ],
        "name": "setName",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

registrar_abi = [
    {
        "inputs": [
            {
                "components": [
                    {"internalType": "string", "name": "name", "type": "string"},
                    {"internalType": "address", "name": "owner", "type": "address"},
                    {"internalType": "uint256", "name": "duration", "type": "uint256"},
                    {"internalType": "address", "name": "resolver", "type": "address"},
                    {"internalType": "bytes[]", "name": "data", "type": "bytes[]"},
                    {"internalType": "bool", "name": "reverseRecord", "type": "bool"}
                ],
                "internalType": "struct RegistrarController.RegisterRequest",
                "name": "request",
                "type": "tuple"
            }
        ],
        "name": "register",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function"
    }
]

