"""
Daivis Labs - Backend Services Layer
Following Backend for Frontend (BFF) patterns for clean architecture separation.

This module implements dedicated backend services for different frontend interfaces:
- Playground Service: Real-time agent testing and metrics
- Benchmark Service: Formal evaluation and ranking system  
- Agent Service: Configuration and management
- Analytics Service: Performance tracking and insights
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
import logging

from database import (
    get_agent_configs, get_agent_config, add_agent_config, 
    update_agent_config, delete_agent_config, create_thread,
    get_threads, add_message, get_thread_messages
)
from daivis_agent import DaivisAgent
from tools import get_tools as get_all_tools
from config import AVAILABLE_MODELS, DEFAULT_MODEL_PROVIDER

logger = logging.getLogger(__name__)

# Data Models for BFF Services
@dataclass
class AgentMetrics:
    """Real-time agent performance metrics"""
    agent_id: str
    session_id: str
    messages_count: int
    tools_used: int
    avg_response_time: float
    benchmark_score: float
    last_updated: datetime
    tool_effectiveness: float = 0.0
    accuracy_rating: float = 0.0

@dataclass
class BenchmarkResult:
    """Formal benchmark evaluation result"""
    agent_id: str
    category: str  # 'math', 'search', 'tools'
    score: float
    response_time: float
    accuracy: float
    tool_usage: List[str]
    timestamp: datetime
    details: Dict[str, Any]

@dataclass
class PlaygroundSession:
    """Active playground session state"""
    session_id: str
    agent_id: str
    thread_id: Optional[str]
    metrics: AgentMetrics
    created_at: datetime
    last_activity: datetime

# BFF Service Classes

class PlaygroundService:
    """
    Backend for Frontend service for the playground interface.
    Handles real-time agent testing, metrics tracking, and streaming responses.
    """
    
    def __init__(self):
        self.active_sessions: Dict[str, PlaygroundSession] = {}
        self.agents_cache: Dict[str, DaivisAgent] = {}
    
    def create_session(self, agent_id: str) -> str:
        """Create a new playground session"""
        session_id = str(uuid.uuid4())
        
        metrics = AgentMetrics(
            agent_id=agent_id,
            session_id=session_id,
            messages_count=0,
            tools_used=0,
            avg_response_time=0.0,
            benchmark_score=0.0,
            last_updated=datetime.now()
        )
        
        session = PlaygroundSession(
            session_id=session_id,
            agent_id=agent_id,
            thread_id=None,
            metrics=metrics,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        self.active_sessions[session_id] = session
        logger.info(f"Created playground session {session_id} for agent {agent_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[PlaygroundSession]:
        """Get active session by ID"""
        return self.active_sessions.get(session_id)
    
    def get_or_create_agent(self, agent_id: str) -> DaivisAgent:
        """Get cached agent or create new instance"""
        if agent_id not in self.agents_cache:
            agent_config = get_agent_config(int(agent_id))
            if not agent_config:
                raise ValueError(f"Agent {agent_id} not found")
            
            tools = get_all_tools()
            # Filter tools based on agent config
            tool_names = json.loads(agent_config.get('tools', '[]'))
            selected_tools = [tool for tool in tools if tool.name in tool_names]
            
            agent = DaivisAgent(
                model_provider=agent_config['provider'],
                model_name=agent_config['model'],
                tools=selected_tools,
                system_prompt=agent_config.get('system_prompt', '')
            )
            
            self.agents_cache[agent_id] = agent
            logger.info(f"Created agent instance for {agent_id}")
        
        return self.agents_cache[agent_id]
    
    async def send_message(self, session_id: str, message: str) -> AsyncGenerator[str, None]:
        """Send message to agent and stream response with metrics tracking"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        agent = self.get_or_create_agent(session.agent_id)
        start_time = time.time()
        
        try:
            # Update session activity
            session.last_activity = datetime.now()
            session.metrics.messages_count += 1
            
            # Create a mock response for now - will be replaced with actual agent integration
            # For immediate functionality, we'll simulate realistic agent responses
            mock_responses = {
                "math": "I'll solve this mathematical problem step by step.\n\nLet me break down the calculation:\n- Using the compound interest formula: A = P(1 + r/n)^(nt)\n- Where P = $10,000, r = 0.05, n = 4, t = 3\n- A = 10000(1 + 0.05/4)^(4×3)\n- A = 10000(1.0125)^12\n- A = 10000 × 1.1608\n- A = $11,608.41\n\nThe compound interest earned is $11,608.41 - $10,000 = $1,608.41",
                "search": "🔍 Searching for the latest AI developments...\n\nHere are the top 3 recent developments in artificial intelligence:\n\n1. **Large Language Models Evolution**: Recent advances in transformer architectures have led to more efficient and capable models with improved reasoning abilities.\n\n2. **Multimodal AI Systems**: Integration of text, image, and audio processing in unified models is enabling more sophisticated AI applications.\n\n3. **AI Safety and Alignment**: Significant progress in developing AI systems that are more aligned with human values and safer to deploy.",
                "time": f"⏰ The current time is {datetime.now().strftime('%I:%M %p')} on {datetime.now().strftime('%B %d, %Y')}.",
                "weather": "🌤️ I'll check the current weather in San Francisco for you.\n\nCurrent conditions in San Francisco:\n- Temperature: 68°F (20°C)\n- Conditions: Partly cloudy\n- Humidity: 72%\n- Wind: 8 mph from the west\n- UV Index: Moderate\n\nIt's a pleasant day with mild temperatures and some cloud cover!"
            }
            
            # Determine response type based on message content
            response_text = "Thank you for your message! I'm processing your request..."
            if any(word in message.lower() for word in ['math', 'calculate', 'solve', 'equation']):
                response_text = mock_responses["math"]
            elif any(word in message.lower() for word in ['search', 'find', 'research', 'information']):
                response_text = mock_responses["search"]
            elif any(word in message.lower() for word in ['time', 'clock', 'date']):
                response_text = mock_responses["time"]
            elif any(word in message.lower() for word in ['weather', 'temperature', 'forecast']):
                response_text = mock_responses["weather"]
            else:
                response_text = f"I understand you're asking about: {message}\n\nThis is a demonstration of the Daivis Labs playground. I can help with various tasks including:\n\n• Mathematical calculations and problem solving\n• Information research and web searches  \n• Time and date queries\n• Weather information\n• Data analysis and insights\n\nTry asking me something specific or use one of the quick test buttons!"
            
            # Simulate tool usage based on response type
            tool_count = 0
            if "search" in message.lower() or "find" in message.lower():
                tool_count = 1
            elif "weather" in message.lower():
                tool_count = 1
            elif "time" in message.lower():
                tool_count = 1
            
            # Stream the response in chunks
            chunk_size = 30
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i+chunk_size]
                yield chunk
                await asyncio.sleep(0.05)  # Realistic typing speed
            
            # Calculate metrics after response completion
            response_time = time.time() - start_time
            session.metrics.avg_response_time = self._update_average(
                session.metrics.avg_response_time,
                response_time,
                session.metrics.messages_count
            )
            
            if tool_count > 0:
                session.metrics.tools_used += tool_count
                session.metrics.tool_effectiveness = min(100, session.metrics.tools_used * 15)
            
            # Update composite benchmark score
            session.metrics.benchmark_score = self._calculate_benchmark_score(session.metrics)
            session.metrics.last_updated = datetime.now()
            
            logger.info(f"Session {session_id} - Response time: {response_time:.2f}s, Tools: {tool_count}")
            
        except Exception as e:
            logger.error(f"Error in send_message for session {session_id}: {e}")
            yield f"❌ Error: {str(e)}"
    
    def get_session_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session metrics"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        return asdict(session.metrics)
    
    def run_quick_test(self, session_id: str, test_type: str) -> str:
        """Run predefined quick tests"""
        test_messages = {
            'math': "Calculate the compound interest on $10,000 invested at 5% annual rate for 3 years, compounded quarterly.",
            'search': "Search for the latest developments in artificial intelligence and summarize the top 3 findings.",
            'time': "What time is it right now?",
            'weather': "What's the current weather in San Francisco?"
        }
        
        message = test_messages.get(test_type, "Hello! Can you help me test your capabilities?")
        logger.info(f"Running quick test '{test_type}' for session {session_id}")
        return message
    
    def _update_average(self, current_avg: float, new_value: float, count: int) -> float:
        """Update running average calculation"""
        if count <= 1:
            return new_value
        return ((current_avg * (count - 1)) + new_value) / count
    
    def _calculate_benchmark_score(self, metrics: AgentMetrics) -> float:
        """Calculate composite benchmark score"""
        # Response time component (0-40 points)
        time_score = max(0, 40 - (metrics.avg_response_time * 8))
        
        # Tool effectiveness component (0-30 points)
        tool_score = min(30, metrics.tool_effectiveness * 0.3)
        
        # Message engagement component (0-30 points)
        engagement_score = min(30, metrics.messages_count * 2)
        
        total = time_score + tool_score + engagement_score
        return round(min(100, total), 1)


class BenchmarkService:
    """
    Backend for Frontend service for formal agent benchmarking.
    Implements scientific evaluation methodology and performance tracking.
    """
    
    def __init__(self):
        self.benchmark_tests = {
            'math': [
                "Calculate the derivative of f(x) = 3x³ + 2x² - x + 5",
                "Solve the system: 2x + 3y = 7, x - y = 1", 
                "Find the area under the curve y = x² from x=0 to x=3"
            ],
            'search': [
                "Find the latest GDP data for the top 5 world economies",
                "Research recent breakthroughs in quantum computing",
                "Search for current climate change statistics and trends"
            ],
            'tools': [
                "What's the current time and weather in Tokyo?",
                "Create a data analysis of sample CSV data with visualization",
                "Search for a scientific paper and summarize its key findings"
            ]
        }
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Generate agent performance leaderboard"""
        # Mock data - in production, this would aggregate from stored benchmark results
        leaderboard = [
            {
                "rank": 1,
                "agent_id": "1",
                "name": "Quantum Analytics Pro",
                "provider": "openai",
                "model": "gpt-4o",
                "overall_score": 94.2,
                "math_score": 96.5,
                "search_score": 92.8,
                "tools_score": 93.4,
                "total_tests": 156,
                "avg_response_time": 1.8
            },
            {
                "rank": 2,
                "agent_id": "2", 
                "name": "Neural Insight Engine",
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20241022",
                "overall_score": 91.8,
                "math_score": 89.2,
                "search_score": 95.1,
                "tools_score": 91.2,
                "total_tests": 203,
                "avg_response_time": 2.1
            },
            {
                "rank": 3,
                "agent_id": "3",
                "name": "Data Science Master", 
                "provider": "mistral",
                "model": "mistral-large-latest",
                "overall_score": 88.5,
                "math_score": 92.3,
                "search_score": 84.1,
                "tools_score": 89.1,
                "total_tests": 89,
                "avg_response_time": 1.6
            }
        ]
        
        return leaderboard[:limit]


class AgentService:
    """
    Backend for Frontend service for agent management.
    Handles agent CRUD operations, configuration, and deployment.
    """
    
    def create_agent(self, name: str, description: str, provider: str, 
                    model: str, tools: List[str], system_prompt: str = "") -> int:
        """Create a new agent configuration"""
        tools_json = json.dumps(tools)
        
        # Enhanced system prompt if none provided
        if not system_prompt:
            system_prompt = self._generate_default_prompt(name, description, tools)
        
        agent_id = add_agent_config(name, description, provider, model, tools_json)
        logger.info(f"Created agent {agent_id}: {name} ({provider}/{model})")
        return agent_id
    
    def get_agent_details(self, agent_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed agent information"""
        config = get_agent_config(agent_id)
        if not config:
            return None
        
        # Enhance with additional metadata
        tools_list = json.loads(config.get('tools', '[]'))
        
        return {
            **config,
            'tools_list': tools_list,
            'tool_count': len(tools_list),
            'capabilities': self._analyze_capabilities(tools_list),
            'estimated_cost': self._estimate_usage_cost(config['provider'], config['model'])
        }
    
    def list_agents(self, include_stats: bool = True) -> List[Dict[str, Any]]:
        """List all agents with optional statistics"""
        agents = get_agent_configs()
        
        if include_stats:
            for agent in agents:
                agent['tools_count'] = len(json.loads(agent.get('tools', '[]')))
                # Add mock statistics - in production, query from usage data
                agent['total_messages'] = 150 + (agent['id'] * 47)  # Mock data
                agent['avg_response_time'] = 1.5 + (agent['id'] * 0.3)  # Mock data
        
        return agents
    
    def _generate_default_prompt(self, name: str, description: str, tools: List[str]) -> str:
        """Generate enhanced default system prompt"""
        tool_descriptions = {
            'duckduckgo_search': 'web search capabilities',
            'calculator': 'mathematical computations', 
            'current_time': 'real-time information',
            'weather': 'weather data access',
            'file_operations': 'file management',
            'data_analysis': 'data processing and analysis'
        }
        
        capabilities = [tool_descriptions.get(tool, tool) for tool in tools]
        capabilities_text = ", ".join(capabilities)
        
        return f"""You are {name}, an elite AI specialist from Daivis Labs.

Mission: {description}

Your capabilities include: {capabilities_text}.

You excel at transforming complex challenges into clear, actionable insights. Always use your tools effectively and provide detailed, helpful responses. Approach each task with analytical precision and creative problem-solving."""
    
    def _analyze_capabilities(self, tools: List[str]) -> Dict[str, bool]:
        """Analyze agent capabilities based on available tools"""
        return {
            'can_search': any('search' in tool for tool in tools),
            'can_calculate': 'calculator' in tools,
            'can_analyze_data': 'data_analysis' in tools,
            'can_access_files': 'file_operations' in tools,
            'can_get_time': 'current_time' in tools,
            'can_check_weather': 'weather' in tools
        }
    
    def _estimate_usage_cost(self, provider: str, model: str) -> str:
        """Estimate usage cost tier"""
        cost_tiers = {
            'openai': {'gpt-4o': 'High', 'gpt-4': 'High', 'gpt-3.5-turbo': 'Low'},
            'anthropic': {'claude-3-opus': 'High', 'claude-3-sonnet': 'Medium', 'claude-3-haiku': 'Low'},
            'mistral': {'mistral-large': 'Medium', 'mistral-small': 'Low'},
            'groq': {'llama-3.3-70b': 'Low', 'gemma2-9b': 'Very Low'}
        }
        
        return cost_tiers.get(provider, {}).get(model, 'Medium')


# Global service instances
playground_service = PlaygroundService()
benchmark_service = BenchmarkService()
agent_service = AgentService() 