"""
Configuration module for the Daivis Labs system.
Intelligence Meets Intuition.
Transform complex data into actionable insights.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq

# Load environment variables from secrets.txt
def load_secrets():
    """Load secrets from secrets.txt file"""
    secrets_path = Path(__file__).parent / "secrets.txt"
    
    if secrets_path.exists():
        with open(secrets_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load secrets
load_secrets()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# LangSmith Configuration (optional for tracing)
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "daivis-labs")

# Model Configurations
DEFAULT_MODEL_PROVIDER = "openai"  # Can be: openai, anthropic, mistral, groq
DEFAULT_MODEL_NAME = "gpt-4o"

MODEL_CONFIGS = {
    "openai": {
        "model": "gpt-4o",  # Latest GPT-4o model
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "anthropic": {
        "model": "claude-3-5-sonnet-20241022",  # Latest Claude 3.5 Sonnet
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "mistral": {
        "model": "mistral-large-latest",  # Latest Mistral Large model
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "groq": {
        "model": "llama-3.3-70b-versatile",  # Updated from deprecated llama3-70b-8192
        "temperature": 0.1,
        "max_tokens": 4096,
    }
}

# Additional model options for each provider
AVAILABLE_MODELS = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini", 
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo"
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022", 
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ],
    "mistral": [
        "mistral-large-latest",
        "mistral-small-latest",
        "mistral-nemo",
        "ministral-3b-latest",
        "ministral-8b-latest"
    ],
    "groq": [
        # Current recommended models (as of 2025)
        "llama-3.3-70b-versatile",  # Replaces deprecated llama3-70b-8192
        "llama-3.1-8b-instant",     # Replaces deprecated llama3-8b-8192
        "meta-llama/llama-4-scout-17b-16e-instruct",  # Latest Llama 4 model
        "meta-llama/llama-4-maverick-17b-128e-instruct",  # Latest Llama 4 model
        "qwen-qwq-32b",             # Reasoning model
        "deepseek-r1-distill-llama-70b",  # DeepSeek reasoning model
        "deepseek-r1-distill-qwen-32b",   # DeepSeek reasoning model
        "gemma2-9b-it",             # Google Gemma model
        "meta-llama/llama-guard-4-12b"  # Content moderation model
    ]
}

def get_llm(provider: str, model_name: str, temperature: float = 0.1, max_tokens: int = 4096, custom_api_keys: dict = None):
    """
    Factory function to get a LangChain LLM instance based on the provider.
    Supports custom API keys when provided.
    """
    # Use custom API key if provided, otherwise fall back to environment/config
    if custom_api_keys:
        api_key = custom_api_keys.get(provider.lower())
    else:
        api_key = None
    
    if provider == "openai":
        return ChatOpenAI(
            api_key=api_key or OPENAI_API_KEY,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            api_key=api_key or ANTHROPIC_API_KEY,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "mistral":
        return ChatMistralAI(
            api_key=api_key or MISTRAL_API_KEY,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "groq":
        return ChatGroq(
            api_key=api_key or GROQ_API_KEY,
            model_name=model_name, # Note: parameter name is 'model_name' for Groq
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unsupported model provider: {provider}")

# Agent Configuration
AGENT_CONFIG = {
    "max_iterations": 8,   # Reduced from 15 to prevent recursion issues
    "verbose": True,
    "return_intermediate_steps": True,
    "handle_parsing_errors": True,
}

# Weather Configuration
WEATHER_CONFIG = {
    "default_units": "metric",  # metric (Celsius), imperial (Fahrenheit), kelvin
    "timeout": 10,              # API request timeout in seconds
    "api_base_url": "http://api.openweathermap.org/data/2.5/weather"
}

# Tool Configuration
SEARCH_CONFIG = {
    "max_results": 5,
    "search_depth": "basic",  # basic, advanced
}

# Memory Configuration
MEMORY_CONFIG = {
    "memory_type": "conversation_summary_buffer",
    "max_token_limit": 2000,
    "return_messages": True,
}

# GraphRAG Configuration
GRAPHRAG_CONFIG = {
    # Neo4j Connection
    "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
    "neo4j_password": os.getenv("NEO4J_PASSWORD", "password"),
    "neo4j_database": os.getenv("NEO4J_DATABASE", "neo4j"),
    
    # Embedding Configuration
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # Lightweight, fast
    "embedding_dimension": 384,
    "chunk_size": 500,
    "chunk_overlap": 50,
    
    # Vector Search Configuration
    "vector_index_name": "skill_embeddings",
    "similarity_threshold": 0.7,
    "max_results": 10,
    
    # Graph Processing
    "enable_community_detection": True,
    "community_algorithm": "louvain",
    "relationship_extraction": True,
    "entity_extraction": True,
    
    # RAG Configuration
    "retrieval_methods": ["vector", "graph_traversal", "hybrid"],
    "rerank_results": True,
    "context_window": 4000,
    "enable_chunk_filtering": True,  # Based on ChunkRAG paper
    
    # Performance
    "batch_size": 100,
    "max_concurrent_requests": 5,
    "cache_embeddings": True,
    "cache_ttl": 3600,  # 1 hour

    # LLM for Extraction (used in Agentic KG Construction)
    "llm_for_extraction_provider": "openai",
    "llm_for_extraction_model": "gpt-4o-mini",
}

# Enhanced Knowledge Graph Configuration
KNOWLEDGE_GRAPH_CONFIG = {
    "data_source": "skills_data_clean.json",
    "enable_neo4j": os.getenv("ENABLE_NEO4J", "false").lower() == "true",
    "auto_migrate": True,
    "create_relationships": True,
    "extract_entities": True,
    "enable_vector_search": True,
    "fallback_to_json": True,  # Fallback to current JSON if Neo4j unavailable
} 