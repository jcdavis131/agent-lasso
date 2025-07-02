"""Simple and reliable tool implementations for the Agentic systems"""

import os
import json
import requests
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

# Import DuckDuckGo search
try:
    from duckduckgo_search import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False
    print("Warning: DuckDuckGo search not available. Install with: pip install duckduckgo-search")


# Input schemas
class SearchInput(BaseModel):
    query: str = Field(description="Search query string")
    max_results: int = Field(default=5, ge=1, le=20)


class FileOperationInput(BaseModel):
    filename: str = Field(description="Name of the file to operate on")
    content: Optional[str] = Field(default=None, description="Content to write to file")
    operation: str = Field(description="Operation: read, write, append, delete, list")


class DataAnalysisInput(BaseModel):
    data_source: str = Field(description="Data source (file path or raw data)")
    analysis_type: str = Field(description="Type: summary, correlation, visualization")
    parameters: Optional[Dict[str, Any]] = Field(default={})


class WeatherInput(BaseModel):
    location: str = Field(description="City name, state/country")
    units: str = Field(default="imperial", description="Temperature units")


class ArxivInput(BaseModel):
    query: str = Field(description="Search query for arXiv papers")
    max_results: int = Field(default=5, ge=1, le=20)
    category: Optional[str] = Field(default=None, description="arXiv category filter")
    sort_by: str = Field(default="relevance", description="Sort results by")


class SearxInput(BaseModel):
    query: str = Field(description="Search query string")
    max_results: int = Field(default=5, ge=1, le=20)
    searx_url: Optional[str] = Field(default="http://localhost:8080")


class InteractiveCanvasInput(BaseModel):
    action: str = Field(description="Canvas action: 'open', 'draw', 'clear', etc.")
    data: Optional[dict] = Field(default=None, description="Optional data for the canvas action")


@tool("duckduckgo_search", args_schema=SearchInput)
def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo - reliable, no API key required."""
    
    if not DUCKDUCKGO_AVAILABLE:
        return """❌ DuckDuckGo search not available. Please install it:
        
🔧 Installation:
pip install duckduckgo-search

📖 Alternative: Use the searx_search tool with local SearXNG"""
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        if not results:
            return f"🔍 No results found for '{query}' on DuckDuckGo"
        
        formatted = []
        formatted.append(f"🦆 **DuckDuckGo Search Results for: '{query}'**")
        formatted.append(f"📊 Found {len(results)} results")
        formatted.append("=" * 80)
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('href', '')
            snippet = result.get('body', 'No description available')
            
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            
            formatted.append(f"\n{i}. {title}")
            formatted.append(f"   🔗 {url}")
            formatted.append(f"   📝 {snippet}")
            
            if i < len(results):
                formatted.append("   " + "-" * 60)
        
        formatted.append("\n" + "=" * 80)
        formatted.append("🦆 Powered by DuckDuckGo - Privacy-focused search")
        
        return "\n".join(formatted)
        
    except Exception as e:
        return f"""❌ DuckDuckGo search failed: {str(e)}

🔄 Alternatives:
1. Try the searx_search tool with local SearXNG
2. Use direct search URLs:
   • DuckDuckGo: https://duckduckgo.com/?q={quote_plus(query)}
   • Google: https://www.google.com/search?q={quote_plus(query)}
   • Bing: https://www.bing.com/search?q={quote_plus(query)}"""


@tool("searx_search", args_schema=SearxInput)
def searx_search(query: str, max_results: int = 5, searx_url: str = "http://localhost:8080") -> str:
    """Search using SearXNG (self-hosted search engine aggregator)."""
    
    try:
        api_url = f"{searx_url}/search"
    
        params = {
            'q': query,
            'format': 'json',
            'engines': 'google,duckduckgo,bing',
            'categories': 'general'
        }
            
        headers = {'User-Agent': 'SOTA-Agent/1.0'}
        
        response = requests.get(api_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = data.get('results', [])
        
        if not results:
            return f"""🔍 No results found for '{query}' on SearXNG

📋 SearXNG Setup Guide:
1. Start local SearXNG with Docker:
   docker run -d -p 8080:8080 searxng/searxng
   
2. Access SearXNG at: http://localhost:8080
   
3. Alternative public instances:
   • https://search.blankenberg.eu
   • https://searx.be
   • https://search.sapti.me"""
        
        results = results[:max_results]
        
        formatted = []
        formatted.append(f"🔍 **SearXNG Search Results for: '{query}'**")
        formatted.append(f"🌐 Instance: {searx_url}")
        formatted.append(f"📊 Found {len(results)} results")
        formatted.append("=" * 80)
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', '')
            content = result.get('content', 'No description available')
            engine = result.get('engine', 'Unknown')
            
            if len(content) > 200:
                content = content[:200] + "..."
            
            formatted.append(f"\n{i}. {title}")
            formatted.append(f"   🔗 {url}")
            formatted.append(f"   🔧 Engine: {engine}")
            formatted.append(f"   📝 {content}")
            
            if i < len(results):
                formatted.append("   " + "-" * 60)
        
        formatted.append("\n" + "=" * 80)
        formatted.append(f"🔍 Powered by SearXNG - Aggregated from multiple search engines")
        
        return "\n".join(formatted)
        
    except requests.ConnectionError:
        return f"""❌ Cannot connect to SearXNG at {searx_url}

🐳 Quick SearXNG Setup with Docker:

1. **Install Docker Desktop** (if not installed):
   • Windows: https://docs.docker.com/desktop/windows/install/
   • Download and run the installer

2. **Start SearXNG container**:
   docker run -d --name searxng -p 8080:8080 searxng/searxng

3. **Verify it's running**:
   • Open browser: http://localhost:8080
   • You should see the SearXNG search interface

4. **Stop/Start container**:
   • Stop: docker stop searxng
   • Start: docker start searxng
   • Remove: docker rm searxng

5. **Alternative: Use public instance**:
   Try changing searx_url to: https://search.blankenberg.eu

💡 Tip: Use duckduckgo_search tool as a reliable alternative!"""
            
    except Exception as e:
        return f"""❌ SearXNG search failed: {str(e)}

🔄 Troubleshooting:
1. Check if SearXNG is running: {searx_url}
2. Try restarting Docker container:
   docker restart searxng
3. Use duckduckgo_search as alternative
4. Try a public SearXNG instance:
   • https://search.blankenberg.eu
   • https://searx.be"""


@tool("file_operations", args_schema=FileOperationInput)
def file_operations(filename: str, content: Optional[str] = None, operation: str = "read") -> str:
    """Perform file operations: read, write, append, delete, or list directory contents."""
    
    try:
        path = Path(filename)
        
        if operation == "read":
            if not path.exists():
                return f"❌ File '{filename}' does not exist"
            
            if path.is_dir():
                return f"❌ '{filename}' is a directory. Use operation 'list' to list contents"
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"📄 Content of '{filename}':\n\n{content}"
            except UnicodeDecodeError:
                with open(path, 'rb') as f:
                    data = f.read()
                return f"📄 Binary file '{filename}' ({len(data)} bytes)\nFirst 200 bytes (hex): {data[:200].hex()}"
        
        elif operation == "write":
            if content is None:
                return "❌ Content is required for write operation"
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"✅ Successfully wrote content to '{filename}'"
        
        elif operation == "append":
            if content is None:
                return "❌ Content is required for append operation"
            
            with open(path, 'a', encoding='utf-8') as f:
                f.write(content)
            return f"✅ Successfully appended content to '{filename}'"
        
        elif operation == "delete":
            if not path.exists():
                return f"❌ File '{filename}' does not exist"
            
            if path.is_dir():
                return f"❌ '{filename}' is a directory. Cannot delete directories for safety"
            
            path.unlink()
            return f"✅ Successfully deleted '{filename}'"
        
        elif operation == "list":
            if path.is_file():
                return f"❌ '{filename}' is a file, not a directory"
            
            if not path.exists():
                path = Path(".")
                filename = "current directory"
            
            items = []
            for item in path.iterdir():
                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f} MB"
                    items.append(f"📄 {item.name} ({size_str})")
            
            if not items:
                return f"📁 Directory '{filename}' is empty"
            
            return f"📁 Contents of '{filename}':\n\n" + "\n".join(sorted(items))
        
        else:
            return f"❌ Unknown operation '{operation}'. Supported: read, write, append, delete, list"
    
    except PermissionError:
        return f"❌ Permission denied accessing '{filename}'"
    except Exception as e:
        return f"❌ Error performing {operation} on '{filename}': {str(e)}"


@tool("data_analysis", args_schema=DataAnalysisInput)
def data_analysis(data_source: str, analysis_type: str, parameters: Dict[str, Any] = {}) -> str:
    """Analyze CSV files or JSON data with summaries, correlations, and visualization preparation."""
    
    try:
        # ------------------------------------------------------------------
        # Import pandas *on demand* – keep the cold-start bundle light
        # ------------------------------------------------------------------
        try:
            import pandas as pd  # noqa: WPS433 – imported inside function intentionally
        except ModuleNotFoundError:
            return (
                "❌ The 'data_analysis' tool requires the 'pandas' package, which "
                "is not installed in this lightweight deployment.  To enable "
                "data analysis features, add `pandas` to your requirements.txt "
                "and redeploy, or run locally with `pip install pandas`."
            )

        if data_source.endswith('.csv') or data_source.endswith('.json'):
            if not Path(data_source).exists():
                return f"❌ Data file '{data_source}' not found"
            
            if data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
            else:
                df = pd.read_json(data_source)
        else:
            try:
                data = json.loads(data_source)
                df = pd.DataFrame(data)
            except json.JSONDecodeError:
                return "❌ Invalid JSON data provided"
        
        if df.empty:
            return "❌ Data source is empty"
        
        if analysis_type == "summary":
            result = []
            result.append(f"📊 Data Summary for: {data_source}")
            result.append("=" * 50)
            result.append(f"📈 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            result.append(f"📋 Columns: {', '.join(df.columns.tolist())}")
            result.append("\n🔢 Data Types:")
            for col, dtype in df.dtypes.items():
                result.append(f"   {col}: {dtype}")
            
            result.append("\n📈 Statistical Summary:")
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary_stats = df[numeric_cols].describe()
                result.append(str(summary_stats))
            else:
                result.append("   No numeric columns found")
            
            result.append("\n❓ Missing Values:")
            missing = df.isnull().sum()
            missing_info = missing[missing > 0]
            if len(missing_info) > 0:
                for col, count in missing_info.items():
                    result.append(f"   {col}: {count} missing ({count/len(df)*100:.1f}%)")
            else:
                result.append("   No missing values found")
            
            return "\n".join(result)
        
        elif analysis_type == "correlation":
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                return "❌ Need at least 2 numeric columns for correlation analysis"
            
            corr_matrix = df[numeric_cols].corr()
            
            result = []
            result.append(f"🔗 Correlation Analysis for: {data_source}")
            result.append("=" * 50)
            result.append("📊 Correlation Matrix:")
            result.append(str(corr_matrix.round(3)))
            
            result.append("\n🔍 Strong Correlations (|r| > 0.7):")
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        strength = "Strong positive" if corr_val > 0 else "Strong negative"
                        strong_corr.append(f"   {col1} ↔ {col2}: {corr_val:.3f} ({strength})")
            
            if strong_corr:
                result.extend(strong_corr)
            else:
                result.append("   No strong correlations found")
            
            return "\n".join(result)
        
        elif analysis_type == "visualization":
            result = []
            result.append(f"📈 Visualization Recommendations for: {data_source}")
            result.append("=" * 50)
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            result.append(f"📊 Data Overview:")
            result.append(f"   Numeric columns: {len(numeric_cols)} ({', '.join(numeric_cols.tolist())})")
            result.append(f"   Categorical columns: {len(categorical_cols)} ({', '.join(categorical_cols.tolist())})")
            
            result.append("\n💡 Suggested Visualizations:")
            
            if len(numeric_cols) >= 2:
                result.append("   📈 Scatter plots for numeric column pairs")
                result.append("   🔗 Correlation heatmap")
            
            if len(numeric_cols) >= 1:
                result.append("   📊 Histograms for distribution analysis")
                result.append("   📦 Box plots for outlier detection")
            
            if len(categorical_cols) >= 1:
                result.append("   🥧 Pie charts for categorical distributions")
                result.append("   📊 Bar charts for category counts")
            
            if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                result.append("   📊 Grouped bar charts (numeric by category)")
                result.append("   📦 Box plots by category")
            
                result.append("\n📋 Sample Data (first 5 rows):")
                result.append(str(df.head()))
            
                return "\n".join(result)
        
            else:
                return f"❌ Unknown analysis type '{analysis_type}'. Supported: summary, correlation, visualization"
    
    except Exception as e:
        return f"❌ Error in data analysis: {str(e)}"


@tool("current_time")
def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"🕐 Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A, %B %d, %Y')})"


@tool("weather", args_schema=WeatherInput)
def get_weather(location: str, units: str = "imperial") -> str:
    """Get current weather information for any location."""
    
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if api_key:
        try:
            base_url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': location,
                'appid': api_key,
                'units': units
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            main = data['main']
            weather = data['weather'][0]
            wind = data.get('wind', {})
            
            temp_unit = "°F" if units == "imperial" else "°C" if units == "metric" else "K"
            speed_unit = "mph" if units == "imperial" else "m/s"
            
            result = []
            result.append(f"🌤️ **Weather for {data['name']}, {data['sys']['country']}**")
            result.append(f"🌡️ Temperature: {main['temp']}{temp_unit} (feels like {main['feels_like']}{temp_unit})")
            result.append(f"📋 Condition: {weather['description'].title()}")
            result.append(f"💧 Humidity: {main['humidity']}%")
            result.append(f"📊 Pressure: {main['pressure']} hPa")
            
            if 'speed' in wind:
                result.append(f"💨 Wind: {wind['speed']} {speed_unit}")
                if 'deg' in wind:
                    result.append(f"🧭 Wind Direction: {wind['deg']}°")
            
            if 'clouds' in data:
                result.append(f"☁️ Cloudiness: {data['clouds']['all']}%")
            
            return "\n".join(result)
            
        except:
            pass
    
    encoded_location = quote_plus(location)
    
    fallback_result = f"""🌤️ **Weather for: {location}**

⚠️ OpenWeatherMap API not available. Get current weather from:

🔗 **Weather Services:**
• Weather.com: https://weather.com/search/enhancedlocalsearch?where={encoded_location}
• AccuWeather: https://www.accuweather.com/en/search-locations?query={encoded_location}
• Weather Underground: https://www.wunderground.com/weather/{encoded_location}
• National Weather Service: https://forecast.weather.gov/

📱 **Mobile Apps:**
• Weather.com app
• AccuWeather app
• Dark Sky (iOS)

🔧 **To enable direct weather data:**
1. Get free API key from: https://openweathermap.org/api
2. Set environment variable: OPENWEATHER_API_KEY=your_key_here
3. Free tier includes 1,000 calls/day"""

    return fallback_result


@tool("calculator")
def calculator(expression: str) -> str:
    """Safely evaluate mathematical expressions using a secure parser."""
    
    try:
        import ast
        import operator
        
        # Define safe operations
        safe_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        
        safe_funcs = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'pow': pow,
        }
        
        def safe_eval(node):
            """Safely evaluate AST node"""
            if isinstance(node, ast.Expression):
                return safe_eval(node.body)
            elif isinstance(node, ast.Constant):  # Numbers
                return node.value
            elif isinstance(node, ast.Num):  # For older Python versions
                return node.n
            elif isinstance(node, ast.BinOp):  # Binary operations
                left = safe_eval(node.left)
                right = safe_eval(node.right)
                op = safe_ops.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
                return op(left, right)
            elif isinstance(node, ast.UnaryOp):  # Unary operations
                operand = safe_eval(node.operand)
                op = safe_ops.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
                return op(operand)
            elif isinstance(node, ast.Call):  # Function calls
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in safe_funcs:
                        args = [safe_eval(arg) for arg in node.args]
                        return safe_funcs[func_name](*args)
                    else:
                        raise ValueError(f"Unsupported function: {func_name}")
                else:
                    raise ValueError("Only simple function names are allowed")
            else:
                raise ValueError(f"Unsupported expression type: {type(node).__name__}")
        
        # Clean the expression
        expression = expression.replace('^', '**').strip()
        
        # Validate characters (more restrictive)
        safe_chars = set('0123456789+-*/().^% abcdefghijklmnopqrstuvwxyz')
        if not all(c.lower() in safe_chars for c in expression):
            return f"❌ Expression contains unsafe characters. Use only numbers, operators, and basic functions."
        
        # Parse and evaluate safely
        try:
            parsed = ast.parse(expression, mode='eval')
            result = safe_eval(parsed)
            
            # Format result
            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(result, 10)  # Limit decimal places
        
            return f"🧮 {expression} = {result}"
            
        except SyntaxError:
            return "❌ Invalid mathematical expression syntax"
    
    except ZeroDivisionError:
        return "❌ Error: Division by zero"
    except ValueError as ve:
        return f"❌ Error: {str(ve)}"
    except Exception as e:
        return f"❌ Error evaluating expression: {str(e)}"


@tool("arxiv_search", args_schema=ArxivInput)
def arxiv_search(query: str, max_results: int = 5, category: Optional[str] = None, sort_by: str = "relevance") -> str:
    """Search for academic papers on arXiv by keywords, authors, categories."""
    
    try:
        base_url = "http://export.arxiv.org/api/query"
        
        search_query = query
        if category:
            search_query = f"cat:{category} AND ({query})"
        
        sort_map = {
            "relevance": "relevance",
            "lastUpdatedDate": "lastUpdatedDate",
            "submittedDate": "submittedDate"
        }
        sort_order = sort_map.get(sort_by, "relevance")
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': sort_order,
            'sortOrder': 'descending'
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        
        entries = root.findall('{http://www.w3.org/2005/Atom}entry')
        
        if not entries:
            search_tips = """
💡 **Search Tips:**
• Use keywords: "machine learning", "neural networks"
• Author search: "au:Hinton", "au:LeCun"
• Title search: "ti:attention mechanism"
• Combine with AND/OR: "machine learning AND reinforcement"
• Use categories: cs.AI, cs.LG, cs.CV, physics.*, math.*
"""
            return f"📚 No papers found for '{query}' in arXiv\n{search_tips}"
        
        formatted = []
        formatted.append(f"📚 **arXiv Search Results for: '{query}'**")
        if category:
            formatted.append(f"🏷️ Category: {category}")
        formatted.append(f"📊 Found {len(entries)} papers (sorted by {sort_by})")
        formatted.append("=" * 80)
        
        for i, entry in enumerate(entries, 1):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            
            authors = []
            for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                name = author.find('{http://www.w3.org/2005/Atom}name').text
                authors.append(name)
            
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
            
            pdf_link = ""
            abs_link = ""
            for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
                if link.get('title') == 'pdf':
                    pdf_link = link.get('href')
                elif link.get('rel') == 'alternate':
                    abs_link = link.get('href')
            
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            pub_date = published.split('T')[0]
            
            categories = []
            for cat in entry.findall('{http://arxiv.org/schemas/atom}primary_category'):
                categories.append(cat.get('term'))
            for cat in entry.findall('{http://arxiv.org/schemas/atom}category'):
                cat_term = cat.get('term')
                if cat_term not in categories:
                    categories.append(cat_term)
            
            formatted.append(f"\n{i}. **{title}**")
            formatted.append(f"   👥 Authors: {', '.join(authors)}")
            formatted.append(f"   📅 Published: {pub_date}")
            if categories:
                formatted.append(f"   🏷️ Categories: {', '.join(categories)}")
            
            if len(summary) > 300:
                summary = summary[:300] + "..."
            formatted.append(f"   📝 Abstract: {summary}")
            
            if abs_link:
                formatted.append(f"   🔗 arXiv Page: {abs_link}")
            if pdf_link:
                formatted.append(f"   📄 PDF: {pdf_link}")
            
            if i < len(entries):
                formatted.append("   " + "-" * 70)
        
        formatted.append("\n" + "=" * 80)
        formatted.append("📚 Powered by arXiv.org - Open access to academic papers")
        
        return "\n".join(formatted)
        
    except requests.RequestException as e:
        return f"❌ Error accessing arXiv API: {str(e)}\n💡 Please check your internet connection and try again."
    except ET.ParseError as e:
        return f"❌ Error parsing arXiv response: {str(e)}\n💡 The arXiv service might be temporarily unavailable."
    except Exception as e:
        return f"❌ Unexpected error searching arXiv: {str(e)}\n💡 Please try again with a different query."


@tool("interactive_canvas", args_schema=InteractiveCanvasInput)
def interactive_canvas(action: str, data: Optional[dict] = None) -> str:
    """Interactive canvas for drawing, sketching, or visualizing ideas. (UI available in playground)"""
    # This is a placeholder for backend logic. Actual interactivity is handled in the playground UI.
    if action == "open":
        return "🖼️ Interactive canvas is now available in the playground UI. Use the canvas to draw, sketch, or visualize ideas."
    elif action == "clear":
        return "🧹 Canvas cleared."
    elif action == "draw":
        return "✏️ Drawing on the canvas (see UI for details)."
    else:
        return f"❓ Unknown canvas action: {action}"


# Tool registry
AVAILABLE_TOOLS = [
    duckduckgo_search,
    searx_search,
    file_operations,
    data_analysis,
    get_current_time,
    get_weather,
    calculator,
    arxiv_search,
    interactive_canvas,
]


def get_tools() -> List[BaseTool]:
    """Return the list of available tools."""
    return AVAILABLE_TOOLS


def get_tool_descriptions() -> str:
    """Return formatted descriptions of all available tools."""
    tool_descriptions = {
        "duckduckgo_search": "Reliable web search using DuckDuckGo (no API key required)",
        "searx_search": "Privacy-focused aggregated search using local/self-hosted SearXNG",
        "file_operations": "Read, write, append, delete files or list directory contents",
        "data_analysis": "Analyze CSV files or JSON data (summary, correlation, visualization prep)",
        "current_time": "Get the current date and time",
        "weather": "Get current weather information for any location (with OpenWeatherMap API or fallback)",
        "calculator": "Safely evaluate mathematical expressions",
        "arxiv_search": "Search academic papers on arXiv by keywords, authors, categories with full details",
        "interactive_canvas": "Interactive canvas for drawing, sketching, or visualizing ideas (UI in playground)",
    }
    
    descriptions = ["Available Tools (Simple & Reliable):"]
    
    for tool in AVAILABLE_TOOLS:
        tool_name = tool.name
        description = tool_descriptions.get(tool_name, tool.description or "No description available")
        descriptions.append(f"- {tool_name}: {description}")
    
    return "\n".join(descriptions) 