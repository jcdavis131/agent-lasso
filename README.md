# Agent Lasso - Configure, Play, Benchmark.

*Transform complex data into actionable insights*

A cutting-edge AI platform engineered for modern professionals who demand both power and elegance. Experience the seamless fusion of advanced analytics, neural intelligence, and visual storytelling—all through an interface designed to inspire.

## 🎯 Philosophy: Elegant Simplicity

This platform embodies:
- ✅ **Zero-Friction Deployment** - Intelligent configuration that just works
- ✅ **Adaptive Intelligence** - Self-optimizing systems with graceful degradation
- ✅ **Privacy by Design** - Your data remains yours, always
- ✅ **Intuitive Mastery** - Complex capabilities through simple interactions
- ✅ **Scalable Architecture** - Built for growth, designed for clarity

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test All Tools
```bash
python quick_tool_test.py
```

### 3. Run the Interface
```bash
# Agent Lasso Interface (Main Application)
python agent_setup_landing.py

# Or Streamlit Interface
streamlit run streamlit_app.py
```

## 🛠️ Available Tools

### Core Tools (Always Available)
- **📊 file_operations** - Read, write, append, delete files or list directories
- **📈 data_analysis** - Analyze CSV/JSON data (summary, correlation, visualization)
- **🕐 current_time** - Get current date and time
- **🧮 calculator** - Safe mathematical expression evaluation

### Search Tools
- **🦆 duckduckgo_search** - Reliable web search (no API key required)
- **🔍 searx_search** - Privacy-focused aggregated search (local SearXNG)

### Optional Enhanced Tools
- **🌤️ weather** - Weather information (with OpenWeatherMap API or fallback links)
- **📚 arxiv_search** - Academic paper search from arXiv

## 🔍 Search Setup

### Option 1: DuckDuckGo (Instant)
Already included - works immediately after installation:
```python
result = duckduckgo_search("Python programming", max_results=5)
```

### Option 2: SearXNG (Privacy-focused)
Set up local SearXNG with Docker for enhanced privacy and aggregated results:

```bash
# Start SearXNG container
docker run -d --name searxng -p 8080:8080 searxng/searxng

# Use in Agent Lasso
result = searx_search("machine learning", max_results=5)
```

**Full setup guide:** [SEARXNG_DOCKER_GUIDE.md](SEARXNG_DOCKER_GUIDE.md)

## 📋 Configuration

### Required: None
The system works without any configuration.

### Optional: Environment Variables

Create a `.env` file for enhanced features:
```bash
# Optional: Enhanced weather data
OPENWEATHER_API_KEY=your_openweather_key

# Optional: Custom SearXNG instance
SEARX_URL=http://localhost:8080
```

## 🧪 Testing

### Quick Test All Tools
```bash
python quick_tool_test.py
```

### Individual Tool Tests
```python
from tools import *

# Test calculator
result = calculator("2 + 3 * 4")
print(result)  # 🧮 2 + 3 * 4 = 14

# Test search
result = duckduckgo_search("Python tutorial", max_results=3)
print(result)  # 🦆 DuckDuckGo Search Results...

# Test data analysis
data = '[{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}]'
result = data_analysis(data, "summary")
print(result)  # 📊 Data Summary...
```

## 🌐 Interfaces

### Agent Lasso Interface (Main Application)
- Modern, responsive UI with glassmorphic design
- Real-time tool execution and AI model configuration
- Advanced chat with thread management
- Complete agent setup and benchmarking tools
- Start: `python agent_setup_landing.py`

### Streamlit Interface
- Simple, clean interface
- Easy to customize
- Start: `streamlit run streamlit_app.py`

### Direct Python Integration
```python
from tools import get_tools

# Get all available tools
tools = get_tools()

# Use with your LLM framework
for tool in tools:
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}")
```

## 📊 System Architecture

```
Agent Lasso (Data + AI + Visualization)
├── Core Tools (No Dependencies)
│   ├── file_operations
│   ├── data_analysis  
│   ├── current_time
│   └── calculator
├── Search Tools
│   ├── duckduckgo_search (Direct)
│   └── searx_search (Local Docker)
├── Enhanced Tools (Optional APIs)
│   ├── weather (OpenWeatherMap + fallback)
│   └── arxiv_search (arXiv API)
└── Interfaces
    ├── FastHTML (Recommended)
    ├── Streamlit
    └── Python Direct
```

## 🔄 Reliability Features

### Automatic Fallbacks
- **Weather Tool**: API → Weather service links
- **Search Tools**: DuckDuckGo → SearXNG → Direct links
- **All Tools**: Comprehensive error handling with helpful guidance

### Error Recovery
```python
# Example: Weather tool with fallback
result = get_weather("New York")
# Returns either:
# 1. Real weather data (if API key configured)
# 2. Links to weather services (fallback mode)
```

### Rate Limiting Protection
- No external rate limits (DuckDuckGo)
- Local SearXNG (unlimited)
- Built-in retry logic

## 📦 Dependencies

### Core (Required)
```
langchain>=0.3.9
langchain-core>=0.3.21
duckduckgo-search>=6.3.4
requests>=2.32.3
pandas>=2.0.0
pydantic>=2.0.0
```

### Interfaces
```
fasthtml>=0.6.9          # For FastHTML interface
streamlit>=1.28.0        # For Streamlit interface
uvicorn>=0.32.0          # Web server
```

### Optional Enhancements
```
docker                   # For SearXNG setup
```

## 🔧 Troubleshooting

### Common Issues

**1. DuckDuckGo Search Not Working**
```bash
pip install --upgrade duckduckgo-search
```

**2. SearXNG Connection Failed**
```bash
# Check if Docker is running
docker ps

# Start SearXNG
docker run -d --name searxng -p 8080:8080 searxng/searxng

# Or use public instance
result = searx_search("query", searx_url="https://search.blankenberg.eu")
```

**3. Tool Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Debugging

Enable debug mode in any interface:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🚀 Advanced Usage

### Custom Tool Integration
```python
from tools import AVAILABLE_TOOLS
from langchain_core.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """My custom tool implementation."""
    return f"Custom result for: {query}"

# Add to available tools
AVAILABLE_TOOLS.append(my_custom_tool)
```

### Search Strategy
```python
def intelligent_search(query: str):
    """Try multiple search approaches."""
    try:
        # Primary: DuckDuckGo
        return duckduckgo_search(query, max_results=5)
    except:
        try:
            # Secondary: Local SearXNG
            return searx_search(query, max_results=5)
        except:
            # Fallback: Public SearXNG
            return searx_search(
                query, 
                max_results=5, 
                searx_url="https://search.blankenberg.eu"
            )
```

## 📝 Changelog

### v2.0 - Simplified & Reliable
- ✅ Removed all API key dependencies for core functionality
- ✅ Added reliable DuckDuckGo search integration
- ✅ Added local SearXNG support with Docker guide
- ✅ Simplified architecture and dependencies
- ✅ Enhanced error handling and fallbacks
- ✅ Updated interfaces for better UX
- ✅ Comprehensive testing and documentation

### v1.0 - Full Featured
- Previous version with extensive API integrations
- Complex rate limiting and proxy management
- Multiple search engine integrations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test with `python quick_tool_test.py`
4. Submit a pull request

## 📄 License

MIT License - feel free to use and modify for your projects.

---

## 🎉 The Future of AI Interaction

Agent Lasso represents a new paradigm in AI platforms—where sophisticated intelligence meets human intuition:

- **Effortless Onboarding** - From installation to insights in minutes
- **Adaptive Intelligence** - Systems that learn and evolve with your workflow  
- **Uncompromising Privacy** - Advanced capabilities without sacrificing data sovereignty
- **Intuitive Interface** - Professional-grade power through elegant simplicity
- **Limitless Potential** - Architecture designed for tomorrow's challenges

Join the next generation of professionals who are transforming how we interact with artificial intelligence.

**Begin Your Journey:** `pip install -r requirements.txt && python agent_setup_landing.py`