import logging
import os
from fastapi import FastAPI, Request, APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import Response as StarletteResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from database import (
    init_db,
    add_agent_config,
    get_agent_configs,
    update_agent_config,
    delete_agent_config,
    add_message,
    get_messages,
    clear_messages,
    create_thread,
    get_threads,
    get_thread,
    update_thread_name,
    delete_thread,
    get_thread_messages,
    clear_thread_messages,
    add_benchmark_result,
    get_recent_benchmarks,
    get_benchmarks_for_agent,
    get_leaderboard,
)
from config import AVAILABLE_MODELS, DEFAULT_MODEL_PROVIDER, GRAPHRAG_CONFIG
from tools import get_tools as get_all_tools
from daivis_agent import DaivisAgent
import json
import uuid
import asyncio
from datetime import datetime
from benchmarks import BENCHMARK_PLUGINS

# Models for request/response validation
class CreateAgentRequest(BaseModel):
    template_key: str = Field(..., description="Template key to use")
    provider: str = Field(default=DEFAULT_MODEL_PROVIDER, description="Model provider")
    model: str = Field(..., description="Model to use")
    custom_name: Optional[str] = Field(None, description="Optional custom name for the agent")
    custom_description: Optional[str] = Field(None, description="Optional custom description")
    custom_system_prompt: Optional[str] = Field(None, description="Optional custom system prompt")
    custom_tools: Optional[List[str]] = Field(None, description="Optional custom tool selection")
    api_keys: Optional[Dict[str, str]] = Field(None, description="API keys for LLM providers")

class AgentResponse(BaseModel):
    id: int
    name: str
    description: str
    provider: str
    model: str
    tools: List[str]
    system_prompt: Optional[str]
    created_at: str

# Additional models for new endpoints
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role (user, assistant)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")

class ChatRequest(BaseModel):
    agent_id: int = Field(..., description="Agent ID to chat with")
    message: str = Field(..., description="User message")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation")

class BenchmarkRequest(BaseModel):
    agent_id: int = Field(..., description="Agent ID to benchmark")
    test_type: str = Field(..., description="Type of benchmark test")
    test_data: Optional[dict] = Field(None, description="Test data")

class BenchmarkResult(BaseModel):
    test_id: str
    agent_id: int
    test_type: str
    score: float
    details: dict
    timestamp: str

# Custom middleware for error handling and logging
class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Unhandled error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "detail": str(e)}
            )

# Environment configuration
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO' if not DEBUG else 'DEBUG')

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security headers
SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' cdn.jsdelivr.net;"
}

# Initialize database
init_db()

# Create FastAPI app
app = FastAPI(
    title="Agent Setup Landing",
    description="AI Agent Configuration Interface",
    version="1.0.0",
    debug=DEBUG
)

# Add middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create router for API endpoints
router = APIRouter(prefix="/api")

# Setup static files
static_path = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_path):
    os.makedirs(static_path)
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Agent Templates System - Pre-configured archetypes with full daivis_agent configuration
AGENT_TEMPLATES = {
    "research_assistant": {
        "name": "Research Assistant",
        "description": "Expert at finding, analyzing, and synthesizing information from multiple sources",
        "emoji": "🔬",
        "category": "Research & Analysis",
        "tools": ["current_time", "duckduckgo_search", "arxiv_search", "calculator", "data_analysis"],
        "knowledge_graphs": ["academic_papers", "research_domains", "citation_networks"],
        "system_prompt": "You are a world-class research assistant with expertise in academic research, data analysis, and information synthesis. Your role is to help users find, analyze, and synthesize information from multiple sources with scientific rigor and academic precision. Always cite sources, provide evidence-based insights, and maintain objectivity in your analysis. You are grounded in the current date and time to provide contextually relevant information.",
        "benchmark_tests": ["arxiv_comprehension", "research_methodology", "data_interpretation"],
        "sample_prompts": [
            "Research the latest developments in quantum computing",
            "Find peer-reviewed papers about climate change impacts",
            "Analyze market trends for renewable energy"
        ],
        "use_cases": ["Academic research", "Market analysis", "Fact-checking"]
    },
    "data_scientist": {
        "name": "Data Scientist",
        "description": "Specialized in data analysis, visualization, and statistical modeling",
        "emoji": "📊",
        "category": "Data & Analytics", 
        "tools": ["current_time", "data_analysis", "calculator", "file_operations"],
        "knowledge_graphs": ["statistical_models", "data_patterns", "ml_algorithms"],
        "system_prompt": "You are an expert data scientist with deep knowledge in statistical analysis, machine learning, and data visualization. Your role is to help users extract insights from data, build predictive models, and create compelling visualizations. Always explain your analytical approach, validate assumptions, and provide actionable recommendations based on data-driven insights. You are grounded in the current date and time to provide contextually relevant analysis.",
        "benchmark_tests": ["statistical_analysis", "ml_model_evaluation", "data_interpretation"],
        "sample_prompts": [
            "Analyze this CSV file and find patterns",
            "Create a statistical model for sales forecasting",
            "Generate visualizations for quarterly metrics"
        ],
        "use_cases": ["Business intelligence", "Predictive modeling", "Data visualization"]
    },
    "creative_writer": {
        "name": "Creative Writer",
        "description": "Crafts engaging content, stories, and marketing copy",
        "emoji": "✍️",
        "category": "Content & Creative",
        "tools": ["current_time", "duckduckgo_search"],
        "knowledge_graphs": ["literary_techniques", "brand_voice", "content_strategies"],
        "system_prompt": "You are a creative writing expert with mastery in storytelling, brand communication, and content creation. Your role is to craft compelling, engaging content that resonates with target audiences. Whether creating stories, marketing copy, or educational content, you blend creativity with strategic thinking to deliver impactful messaging. You are grounded in the current date and time to ensure your content is timely and relevant.",
        "benchmark_tests": ["creative_writing", "brand_consistency", "content_engagement"],
        "sample_prompts": [
            "Write a compelling product description",
            "Create a short story about space exploration",
            "Draft a marketing email for our new product"
        ],
        "use_cases": ["Content marketing", "Creative writing", "Copywriting"]
    },
    "tech_support": {
        "name": "Tech Support Specialist",
        "description": "Troubleshoots technical issues and provides step-by-step solutions",
        "emoji": "🛠️",
        "category": "Technical Support",
        "tools": ["current_time", "duckduckgo_search", "file_operations"],
        "knowledge_graphs": ["tech_troubleshooting", "hardware_compatibility", "software_solutions"],
        "system_prompt": "You are a patient and knowledgeable technical support specialist with expertise in diagnosing and resolving technical issues. Your role is to provide clear, step-by-step guidance that helps users solve problems efficiently. Always ask clarifying questions, explain technical concepts in simple terms, and ensure users feel confident with the solutions. You are grounded in the current date and time to provide up-to-date technical guidance.",
        "benchmark_tests": ["problem_diagnosis", "solution_effectiveness", "user_satisfaction"],
        "sample_prompts": [
            "Help me troubleshoot my WiFi connection",
            "Explain how to backup my computer files",
            "Fix this error message I'm getting"
        ],
        "use_cases": ["Customer support", "IT helpdesk", "Technical guidance"]
    },
    "personal_assistant": {
        "name": "Personal Assistant",
        "description": "Helps with scheduling, reminders, and daily task management",
        "emoji": "🗓️",
        "category": "Productivity",
        "tools": ["current_time", "weather", "duckduckgo_search"],
        "knowledge_graphs": ["productivity_systems", "time_management", "lifestyle_optimization"],
        "system_prompt": "You are a highly organized and proactive personal assistant focused on optimizing productivity and work-life balance. Your role is to help users manage their time effectively, stay organized, and achieve their goals. You're attentive to detail, anticipate needs, and provide practical solutions for daily challenges. You are always aware of the current date and time to provide timely assistance.",
        "benchmark_tests": ["task_prioritization", "schedule_optimization", "goal_achievement"],
        "sample_prompts": [
            "What's the weather like today?",
            "Help me plan my weekly schedule",
            "Find restaurants near me for dinner"
        ],
        "use_cases": ["Personal productivity", "Schedule management", "Daily assistance"]
    },
    "financial_advisor": {
        "name": "Financial Advisor",
        "description": "Provides insights on investments, budgeting, and financial planning",
        "emoji": "💰",
        "category": "Finance & Business",
        "tools": ["current_time", "calculator", "duckduckgo_search", "data_analysis"],
        "knowledge_graphs": ["financial_markets", "investment_strategies", "economic_indicators"],
        "system_prompt": "You are a trusted financial advisor with deep expertise in investment planning, budgeting, and wealth management. Your role is to provide sound financial guidance tailored to individual circumstances and goals. Always consider risk tolerance, time horizons, and regulatory compliance while explaining financial concepts clearly. You are grounded in the current date and time to provide timely market insights and financial advice.",
        "benchmark_tests": ["financial_analysis", "investment_recommendations", "risk_assessment"],
        "sample_prompts": [
            "Calculate compound interest for my savings",
            "Research current stock market trends",
            "Help me create a monthly budget"
        ],
        "use_cases": ["Investment planning", "Budget management", "Financial analysis"]
    }
}

def ensure_current_time_tool(tools_list):
    """Ensure current_time tool is always included in the tools list for temporal grounding"""
    if isinstance(tools_list, str):
        tools_list = json.loads(tools_list)
    
    # Ensure current_time is always the first tool for temporal grounding
    if "current_time" not in tools_list:
        tools_list.insert(0, "current_time")
    elif tools_list.index("current_time") != 0:
        # Move current_time to the front if it's not already there
        tools_list.remove("current_time")
        tools_list.insert(0, "current_time")
    
    return tools_list

def render_md(text):
    """Simple markdown-like rendering for basic formatting"""
    if not text:
        return ""
    
    text = text.replace('**', '<strong>', 1)
    if '<strong>' in text:
        text = text.replace('**', '</strong>', 1)
    
    text = text.replace('*', '<em>', 1)
    if '<em>' in text and text.count('*') > 0:
        text = text.replace('*', '</em>', 1)
    
    text = text.replace('\n', '<br>')
    
    return text

# Error handling utility
def handle_tools_reconstruction(tools, available_tool_names):
    """Handle broken form submission tool reconstruction"""
    if not tools or len(tools) == 0:
        return []
    
    # Check if we have individual characters (broken form submission)
    if all(len(str(tool)) == 1 for tool in tools):
        logger.info("Detected broken form submission - reconstructing tool names")
        tool_string = ''.join(tools)
        
        # Try to split by known tool names
        reconstructed_tools = []
        current_tool = ""
        
        for char in tool_string:
            current_tool += char
            if current_tool in available_tool_names:
                reconstructed_tools.append(current_tool)
                current_tool = ""
                
        return reconstructed_tools
    
    return tools

# Dependency for template validation
async def validate_template(request: CreateAgentRequest) -> dict:
    if request.template_key not in AGENT_TEMPLATES:
        raise HTTPException(status_code=400, detail="Invalid template key")
    return AGENT_TEMPLATES[request.template_key]

# Dependency for model validation
async def validate_model(request: CreateAgentRequest) -> tuple:
    if request.provider not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Invalid provider")
    if request.model not in AVAILABLE_MODELS[request.provider]:
        raise HTTPException(status_code=400, detail="Invalid model for provider")
    return request.provider, request.model

# Main route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, chat: str = None):
    """Render the main application page"""
    try:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "chat": chat,
                "agent_templates": AGENT_TEMPLATES,
                "available_models": AVAILABLE_MODELS,
                "default_provider": DEFAULT_MODEL_PROVIDER
            }
        )
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading page")

# API routes
@router.get("/tools")
async def get_available_tools():
    """Get detailed list of available tools with descriptions"""
    try:
        tools = get_all_tools()
        
        # Enhanced tool information with categories and descriptions
        tool_info = {
            "duckduckgo_search": {
                "name": "duckduckgo_search",
                "display_name": "DuckDuckGo Search",
                "description": "Reliable web search using DuckDuckGo (privacy-focused, no API key required)",
                "category": "Research & Search",
                "icon": "fas fa-search",
                "required": False
            },
            "searx_search": {
                "name": "searx_search", 
                "display_name": "SearXNG Search",
                "description": "Privacy-focused aggregated search using self-hosted SearXNG",
                "category": "Research & Search",
                "icon": "fas fa-search-plus",
                "required": False
            },
            "arxiv_search": {
                "name": "arxiv_search",
                "display_name": "arXiv Search", 
                "description": "Search academic papers on arXiv by keywords, authors, and categories",
                "category": "Research & Search",
                "icon": "fas fa-graduation-cap",
                "required": False
            },
            "calculator": {
                "name": "calculator",
                "display_name": "Calculator",
                "description": "Safely evaluate mathematical expressions and calculations",
                "category": "Math & Computation",
                "icon": "fas fa-calculator",
                "required": False
            },
            "data_analysis": {
                "name": "data_analysis",
                "display_name": "Data Analysis",
                "description": "Analyze CSV files or JSON data (summary, correlation, visualization)",
                "category": "Data & Analytics",
                "icon": "fas fa-chart-bar",
                "required": False
            },
            "file_operations": {
                "name": "file_operations",
                "display_name": "File Operations", 
                "description": "Read, write, append, delete files or list directory contents",
                "category": "File & System",
                "icon": "fas fa-file-alt",
                "required": False
            },
            "weather": {
                "name": "weather",
                "display_name": "Weather",
                "description": "Get current weather information for any location worldwide",
                "category": "Information & Utilities",
                "icon": "fas fa-cloud-sun",
                "required": False
            },
            "current_time": {
                "name": "current_time",
                "display_name": "Current Time",
                "description": "Get the current date and time (automatically included for temporal grounding)",
                "category": "Information & Utilities", 
                "icon": "fas fa-clock",
                "required": True
            },
            "interactive_canvas": {
                "name": "interactive_canvas",
                "display_name": "Interactive Canvas",
                "description": "Interactive canvas for drawing, sketching, or visualizing ideas",
                "category": "Creative & Visual",
                "icon": "fas fa-paint-brush",
                "required": False
            }
        }
        
        # Build the response with detailed tool information
        detailed_tools = []
        available_tool_names = [tool.name for tool in tools]
        
        for tool_name in available_tool_names:
            if tool_name in tool_info:
                detailed_tools.append(tool_info[tool_name])
            else:
                # Fallback for any tools not in our predefined info
                detailed_tools.append({
                    "name": tool_name,
                    "display_name": tool_name.replace('_', ' ').title(),
                    "description": f"Tool: {tool_name}",
                    "category": "Other",
                    "icon": "fas fa-cog",
                    "required": False
                })
        
        # Group tools by category
        categories = {}
        for tool in detailed_tools:
            category = tool["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(tool)
        
        return {
            "tools": detailed_tools,
            "categories": categories,
            "total_count": len(detailed_tools)
        }
    except Exception as e:
        logger.error(f"Error getting tools: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get tools")

@router.post("/create_agent_from_template")
async def create_agent_from_template(
    request: CreateAgentRequest,
    template: dict = Depends(validate_template),
    validated_model: tuple = Depends(validate_model)
):
    """Create a new agent from a template"""
    try:
        provider, model = validated_model
        
        # Use custom tools if provided, otherwise use template tools
        if request.custom_tools is not None:
            tools_list = request.custom_tools
        else:
            tools_list = template.get("tools", [])
        
        # Ensure current_time tool is always included for temporal grounding
        tools_list = ensure_current_time_tool(tools_list)
        tools_json = json.dumps(tools_list)
        
        # Use custom values if provided, otherwise use template defaults
        name = request.custom_name or f"{template['name']} ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        description = request.custom_description or template["description"]
        system_prompt = request.custom_system_prompt or template.get("system_prompt", "")
        
        # Handle API keys - store as JSON if provided
        api_keys_json = None
        if request.api_keys:
            # Validate that API key is provided for the selected provider
            if provider.lower() not in request.api_keys:
                raise HTTPException(
                    status_code=400, 
                    detail=f"API key for {provider} is required when using this provider"
                )
            api_keys_json = json.dumps(request.api_keys)
        
        # Add agent to database
        agent_id = add_agent_config(
            name=name,
            description=description,
            provider=provider,
            model=model,
            tools_json=tools_json,
            system_prompt=system_prompt,
            api_keys=api_keys_json
        )
        
        return {
            "agent_id": agent_id,
            "message": "Agent created successfully",
            "template": request.template_key
        }
    except Exception as e:
        logger.error(f"Error creating agent from template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents")
async def list_agents():
    """Get list of all agents"""
    try:
        agents = get_agent_configs()
        # Transform the data to match the response model
        transformed_agents = []
        for agent in agents:
            # Parse tools and ensure current_time is included for temporal grounding
            tools_list = json.loads(agent["tools"]) if isinstance(agent["tools"], str) else agent["tools"]
            tools_list = ensure_current_time_tool(tools_list)
            
            # Fetch latest benchmark (if any) for this agent
            latest_benchmark = None
            try:
                agent_history = get_benchmarks_for_agent(agent["id"], limit=1)
                latest_benchmark = agent_history[0] if agent_history else None
            except Exception as hist_err:
                logger.debug(f"Could not fetch benchmark history for agent {agent['id']}: {hist_err}")
            
            transformed_agent = {
                "id": agent["id"],
                "name": agent["name"],
                "description": agent.get("description", ""),
                "provider": agent["provider"],
                "model": agent["model"],
                "tools": tools_list,
                "system_prompt": agent.get("system_prompt", ""),
                "created_at": agent["created_at"],
                "latest_benchmark": latest_benchmark,
            }
            transformed_agents.append(transformed_agent)
        return {"agents": transformed_agents}
    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list agents")

@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: int):
    """Delete an agent configuration"""
    try:
        delete_agent_config(agent_id)
        return {"message": "Agent deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/agents/{agent_id}")
async def update_agent(
    agent_id: int,
    request: CreateAgentRequest,
    validated_model: tuple = Depends(validate_model)
):
    """Update an existing agent configuration"""
    try:
        provider, model = validated_model
        
        # Get existing agent to preserve unchanged values
        existing_agent = get_agent_configs(agent_id)
        if not existing_agent:
            raise HTTPException(status_code=404, detail="Agent not found")
            
        # Update agent in database
        update_agent_config(
            agent_id=agent_id,
            name=request.custom_name or existing_agent["name"],
            description=request.custom_description or existing_agent["description"],
            provider=provider,
            model=model,
            tools_json=existing_agent["tools"],
            system_prompt=request.custom_system_prompt or existing_agent.get("system_prompt", "")
        )
        
        return {"message": "Agent updated successfully"}
    except Exception as e:
        logger.error(f"Error updating agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """Chat with a specific agent"""
    try:
        # Get agent configuration
        agents = get_agent_configs()
        agent = next((a for a in agents if a["id"] == request.agent_id), None)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Create or get thread
        thread_id = request.thread_id or str(uuid.uuid4())
        current_time = datetime.now().isoformat()
        
        # Initialize agent with correct parameters and ensure current_time tool is included
        tools_list = json.loads(agent["tools"]) if isinstance(agent["tools"], str) else agent["tools"]
        tools_list = ensure_current_time_tool(tools_list)  # Ensure temporal grounding
        
        # Get custom API keys if available
        custom_api_keys = None
        if agent.get("api_keys"):
            custom_api_keys = json.loads(agent["api_keys"])
        
        daivis_agent = DaivisAgent(
            model_provider=agent["provider"],
            model_name=agent["model"],
            tools_config=tools_list,
            system_prompt=agent.get("system_prompt", ""),
            custom_api_keys=custom_api_keys
        )
        
        # Get conversation history
        messages = get_thread_messages(thread_id) if request.thread_id else []
        
        # Add user message to history
        add_message(thread_id, "user", request.message, current_time)
        
        # Generate response using the correct method
        response = daivis_agent.run(request.message, thread_id)
        
        # Add assistant response to history
        add_message(thread_id, "assistant", response, current_time)
        
        return {
            "response": response,
            "thread_id": thread_id,
            "agent_id": request.agent_id,
            "timestamp": current_time
        }
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_with_agent_stream(request: ChatRequest):
    """Stream chat with a specific agent with progress indicators"""
    
    async def generate_stream():
        try:
            # Get agent configuration
            agents = get_agent_configs()
            agent = next((a for a in agents if a["id"] == request.agent_id), None)
            if not agent:
                yield f"data: {json.dumps({'type': 'error', 'content': 'Agent not found'})}\n\n"
                return
            
            # Create or get thread
            thread_id = request.thread_id or str(uuid.uuid4())
            current_time = datetime.now().isoformat()
            
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'content': 'Initializing agent...', 'thread_id': thread_id})}\n\n"
            
            # Initialize agent with correct parameters
            tools_list = json.loads(agent["tools"]) if isinstance(agent["tools"], str) else agent["tools"]
            tools_list = ensure_current_time_tool(tools_list)
            
            # Get custom API keys if available
            custom_api_keys = None
            if agent.get("api_keys"):
                custom_api_keys = json.loads(agent["api_keys"])
            
            daivis_agent = DaivisAgent(
                model_provider=agent["provider"],
                model_name=agent["model"], 
                tools_config=tools_list,
                system_prompt=agent.get("system_prompt", ""),
                custom_api_keys=custom_api_keys
            )
            
            # Add user message to history
            add_message(thread_id, "user", request.message, current_time)
            
            yield f"data: {json.dumps({'type': 'thinking', 'content': 'Processing your request...'})}\n\n"
            
            # Set up session state manually for progress tracking
            session_id = thread_id
            daivis_agent.session_states[session_id] = {
                "messages": [{"role": "system", "content": agent.get("system_prompt", "")}, {"role": "user", "content": request.message}],
                "tool_invocations": [],
                "retrieved_context": []
            }
            
            # Step 1: Call LLM for initial response
            yield f"data: {json.dumps({'type': 'thinking', 'content': 'Agent is thinking...'})}\n\n"
            
            # Run the agent with progress updates
            response = ""
            async for progress_message in run_agent_with_progress_generator(daivis_agent, request.message, thread_id):
                if progress_message.startswith("FINAL_RESPONSE:"):
                    response = progress_message[15:]  # Remove "FINAL_RESPONSE:" prefix
                else:
                    yield progress_message
            
            # Add assistant response to history
            add_message(thread_id, "assistant", response, current_time)
            
            # Send final response
            yield f"data: {json.dumps({'type': 'response', 'content': response, 'timestamp': current_time})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming chat: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

async def run_agent_with_progress_generator(agent, message, thread_id):
    """Run agent with progress updates as an async generator"""
    import asyncio
    
    try:
        # Initialize session state
        from langchain_core.messages import HumanMessage, AIMessage
        
        session_state = {
            "messages": [HumanMessage(content=message)],
            "tool_invocations": [],
            "retrieved_context": []
        }
        
        # Call LLM for initial response
        yield f"data: {json.dumps({'type': 'thinking', 'content': 'Analyzing your request...'})}\n\n"
        await asyncio.sleep(0.5)  # Small delay for UX
        
        llm_output = agent.call_model(session_state)
        session_state["messages"].extend(llm_output["messages"])
        
        # Check if tools need to be called
        last_message = session_state["messages"][-1]
        if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Tools need to be called
            yield f"data: {json.dumps({'type': 'tool_use', 'content': 'Using tools to gather information...'})}\n\n"
            
            tool_names = [tc['name'] for tc in last_message.tool_calls]
            tool_names_str = ', '.join(tool_names)
            yield f"data: {json.dumps({'type': 'tool_use', 'content': f'Calling tools: {tool_names_str}'})}\n\n"
            
            # Execute tools
            tool_output = agent.tool_node(session_state)
            session_state["tool_invocations"].extend(tool_output["tool_invocations"])
            
            yield f"data: {json.dumps({'type': 'thinking', 'content': 'Processing tool results...'})}\n\n"
            await asyncio.sleep(0.5)
            
            # Call LLM again to interpret tool output
            llm_output = agent.call_model(session_state)
            session_state["messages"].extend(llm_output["messages"])
        
        # Get final answer
        final_answer = session_state["messages"][-1].content
        
        # Apply refinement if available
        if agent.refinement_agent:
            yield f"data: {json.dumps({'type': 'thinking', 'content': 'Refining response...'})}\n\n"
            
            refinement_state = {
                "messages": session_state["messages"],
                "original_answer": final_answer,
                "retrieved_context": session_state.get('retrieved_context', [])
            }
            
            try:
                refined_answer = await agent.refinement_agent.refine_answer(refinement_state)
                if refined_answer != final_answer:
                    final_answer = refined_answer
            except Exception as e:
                logger.error(f"Refinement failed: {e}")
        
        # Return the final answer with a special prefix
        yield f"FINAL_RESPONSE:{final_answer}"
        
    except Exception as e:
        logger.error(f"Error in run_agent_with_progress_generator: {e}")
        yield f"FINAL_RESPONSE:Error processing request: {str(e)}"

@router.get("/chat/threads")
async def get_chat_threads():
    """Get all chat threads"""
    try:
        threads = get_threads()
        return {"threads": threads}
    except Exception as e:
        logger.error(f"Error getting threads: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get threads")

@router.get("/chat/threads/{thread_id}/messages")
async def get_thread_messages_endpoint(thread_id: str):
    """Get messages for a specific thread"""
    try:
        messages = get_thread_messages(thread_id)
        return {"messages": messages, "thread_id": thread_id}
    except Exception as e:
        logger.error(f"Error getting thread messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get messages")

@router.post("/benchmark")
async def run_benchmark(request: BenchmarkRequest):
    """Run benchmark tests on an agent using pluggable harnesses"""
    try:
        # Validate benchmark type
        if request.test_type not in BENCHMARK_PLUGINS:
            raise HTTPException(status_code=400, detail="Invalid test type")

        # Retrieve agent configuration
        agents = get_agent_configs()
        agent = next((a for a in agents if a["id"] == request.agent_id), None)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        test_id = str(uuid.uuid4())

        # Build DaivisAgent instance with ensured temporal grounding
        tools_list = json.loads(agent["tools"]) if isinstance(agent["tools"], str) else agent["tools"]
        tools_list = ensure_current_time_tool(tools_list)

        daivis_agent = DaivisAgent(
            model_provider=agent["provider"],
            model_name=agent["model"],
            tools_config=tools_list,
            system_prompt=agent.get("system_prompt", ""),
        )

        # Execute selected benchmark harness
        harness = BENCHMARK_PLUGINS[request.test_type]
        harness_result = harness.run(daivis_agent, test_id)

        # Standardise response
        result = {
            "test_id": test_id,
            "agent_id": request.agent_id,
            "test_type": request.test_type,
            "score": harness_result.get("score"),
            "details": harness_result.get("details", {}),
            "timestamp": datetime.now().isoformat(),
        }

        # Persist result for history/leaderboards
        try:
            add_benchmark_result(
                id_=test_id,
                agent_id=request.agent_id,
                test_type=request.test_type,
                score=result["score"],
                details=result["details"],
            )
        except Exception as db_err:
            logger.warning(f"Failed to persist benchmark result {test_id}: {db_err}")

        return result

    except HTTPException:
        # Re-raise explicit HTTP errors untouched
        raise
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/benchmark/history/{agent_id}")
async def get_benchmark_history(agent_id: int):
    """Get persistent benchmark history for an agent"""
    try:
        history = get_benchmarks_for_agent(agent_id, limit=50)
        return {"agent_id": agent_id, "benchmarks": history}
    except Exception as e:
        logger.error(f"Error getting benchmark history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/{agent_id}")
async def get_agent(agent_id: int):
    """Get a specific agent configuration"""
    try:
        agents = get_agent_configs()
        agent = next((a for a in agents if a["id"] == agent_id), None)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Transform the data
        transformed_agent = {
            "id": agent["id"],
            "name": agent["name"],
            "description": agent.get("description", ""),
            "provider": agent["provider"],
            "model": agent["model"],
            "tools": json.loads(agent["tools"]) if isinstance(agent["tools"], str) else agent["tools"],
            "system_prompt": agent.get("system_prompt", ""),
            "created_at": agent["created_at"]
        }
        return transformed_agent
    except Exception as e:
        logger.error(f"Error getting agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/benchmark/types")
async def list_benchmark_types():
    """Return available benchmark identifiers and descriptions for the UI dropdown."""
    try:
        benchmarks = [
            {"key": key, "description": harness.description} for key, harness in BENCHMARK_PLUGINS.items()
        ]
        # Sort alphabetically for readability
        benchmarks.sort(key=lambda x: x["key"])
        return {"benchmarks": benchmarks, "count": len(benchmarks)}
    except Exception as e:
        logger.error(f"Error listing benchmarks: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list benchmark types")

@router.get("/benchmark/stream")
async def benchmark_stream(agent_id: int, test_type: str):
    """Stream benchmark execution step-by-step via SSE."""
    if test_type not in BENCHMARK_PLUGINS:
        raise HTTPException(status_code=400, detail="Invalid test type")

    # get agent
    agents = get_agent_configs()
    agent = next((a for a in agents if a["id"] == agent_id), None)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    test_id = str(uuid.uuid4())

    tools_list = ensure_current_time_tool(json.loads(agent["tools"]) if isinstance(agent["tools"], str) else agent["tools"])
    daivis_agent = DaivisAgent(
        model_provider=agent["provider"],
        model_name=agent["model"],
        tools_config=tools_list,
        system_prompt=agent.get("system_prompt", ""),
    )

    harness = BENCHMARK_PLUGINS[test_type]

    async def event_generator():
        for msg in harness.run_iter(daivis_agent, test_id):
            yield f"event:{msg['event']}\ndata:{json.dumps(msg['data'])}\n\n"
            if msg['event'] == 'final':
                # Save to DB
                add_benchmark_result(test_id, agent_id, test_type, msg['data'].get('score',0.0), msg['data'])

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/benchmark/history/latest")
async def latest_benchmarks(limit: int = 5):
    try:
        return {"results": get_recent_benchmarks(limit)}
    except Exception as e:
        logger.error(f"Error fetching benchmark history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")

@router.get("/benchmark/leaderboard")
async def benchmark_leaderboard(limit: int = 10):
    """Return top agents ranked by average benchmark score."""
    try:
        return {"leaderboard": get_leaderboard(limit)}
    except Exception as e:
        logger.error(f"Error fetching leaderboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch leaderboard")

# Include router
app.include_router(router)

if __name__ == "__main__":
    print("Starting Agent Lasso...")