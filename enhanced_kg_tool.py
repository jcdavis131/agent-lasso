"""
Enhanced Knowledge Graph Tool for Daivis Labs Agent
Integrates GraphRAG engine with LangChain tool interface
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from langchain_core.tools import BaseTool
from pydantic import Field
from langchain_core.documents import Document

from graphrag_engine import get_graph_rag_instance, BaseGraphRAG, ChunkFilteringLLM
from config import GRAPHRAG_CONFIG, KNOWLEDGE_GRAPH_CONFIG

logger = logging.getLogger(__name__)

class EnhancedKnowledgeGraphTool(BaseTool):
    """
    Enhanced Knowledge Graph Tool with GraphRAG capabilities:
    - Vector similarity search
    - Graph traversal search  
    - Hybrid retrieval methods
    - LLM-based chunk filtering
    - Contextual result formatting
    """
    
    name: str = "enhanced_knowledge_graph"
    description: str = (
        "Query the enhanced knowledge graph using advanced retrieval methods. "
        "Supports vector similarity search, graph traversal, and hybrid approaches. "
        "Input format: 'method:query' where method can be 'vector', 'graph', or 'hybrid'. "
        "If no method specified, defaults to hybrid search. "
        "Examples: 'hybrid:machine learning', 'vector:python programming', 'graph:Microsoft'"
    )
    
    # Class-level engine instance
    _engine: Optional[BaseGraphRAG] = None
    _initialization_lock = asyncio.Lock()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize engine lazily
        if EnhancedKnowledgeGraphTool._engine is None:
            # Use a lock to prevent multiple initializations in a multi-threaded environment
            if not EnhancedKnowledgeGraphTool._initialization_lock.locked():
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self._async_initialize_engine())
                        future.result()
                else:
                    asyncio.run(self._async_initialize_engine())

    async def _async_initialize_engine(self):
        async with EnhancedKnowledgeGraphTool._initialization_lock:
            if EnhancedKnowledgeGraphTool._engine is None:
                try:
                    EnhancedKnowledgeGraphTool._engine = get_graph_rag_instance(GRAPHRAG_CONFIG)
                    logger.info("GraphRAG engine initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize GraphRAG engine: {e}")
                    EnhancedKnowledgeGraphTool._engine = None
    
    def _parse_query(self, query: str) -> tuple[str, str]:
        """Parse query to extract method and search terms"""
        if ':' in query and query.split(':', 1)[0].lower() in ['vector', 'graph', 'hybrid']:
            method, search_query = query.split(':', 1)
            return method.lower().strip(), search_query.strip()
        else:
            return 'hybrid', query.strip()
    
    def _format_results(self, results: List[Document], query: str, method: str) -> str:
        """Format retrieval results for display"""
        if not results:
            return f"No results found for '{query}' using {method} search method.\n\nTip: Try different search terms or methods (vector, graph, hybrid)."
        
        formatted_parts = [f"🔍 **Enhanced Knowledge Graph Results for '{query}'** (Method: {method.title()})\n"]
        
        for i, result in enumerate(results[:5], 1):
            content = result.page_content
            source = result.metadata.get('source', 'Unknown Source')
            score = result.metadata.get('score', 'N/A')
            name = result.metadata.get('name', f"Result {i}")
            
            formatted_parts.append(f"{i}. **{name}** (Score: {score:.2f} if isinstance(score, float) else score)")
            formatted_parts.append(f"   📍 Source: {source}")
            formatted_parts.append(f"   📝 {content[:200]}{'...' if len(content) > 200 else ''}")
            if method == "hybrid":
                formatted_parts.append(f"   🔬 Method: Combined vector + graph search")
            formatted_parts.append("")
        
        method_explanations = {
            "vector": "🧠 Vector search uses semantic similarity to find conceptually related content",
            "graph": "🕸️ Graph search traverses relationships to find structurally connected content", 
            "hybrid": "⚡ Hybrid search combines vector similarity with graph traversal for comprehensive results"
        }
        
        if method in method_explanations:
            formatted_parts.append(f"💡 **About {method.title()} Search:** {method_explanations[method]}")
        
        formatted_parts.append("\n🔧 **Search Tips:**")
        formatted_parts.append("• Use 'vector:query' for conceptual/semantic searches")
        formatted_parts.append("• Use 'graph:query' for exact matches and related categories")
        formatted_parts.append("• Use 'hybrid:query' (default) for comprehensive results")
        formatted_parts.append("• Try different keywords if results aren't relevant")
        
        return "\n".join(formatted_parts)
    
    async def _async_search(self, query: str) -> List[Document]:
        """
        Async search implementation - returns raw Document list
        """
        if not query.strip():
            return []
        
        if not EnhancedKnowledgeGraphTool._engine:
            logger.warning("GraphRAG engine not available for async search.")
            return []
        
        try:
            method, search_query = self._parse_query(query)
            
            results = EnhancedKnowledgeGraphTool._engine.retrieve_relevant_context(query=search_query)
            
            if EnhancedKnowledgeGraphTool._engine.enable_chunk_filtering and hasattr(EnhancedKnowledgeGraphTool._engine, 'llm_for_extraction'):
                chunk_filter_llm = EnhancedKnowledgeGraphTool._engine.llm_for_extraction
                chunk_filter = ChunkFilteringLLM(llm_for_filtering=chunk_filter_llm)
                results = chunk_filter.filter_chunks(search_query, results)

            return results
            
        except Exception as e:
            logger.error(f"GraphRAG search failed: {e}")
            return []

    def invoke(self, query: str) -> str:
        """
        Synchronous invoke method - wraps async search and formats results.
        This is primarily for Langchain tool invocation where string output is expected.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._async_search(query))
                    raw_results = future.result()
            else:
                raw_results = asyncio.run(self._async_search(query))
            
            method, search_query = self._parse_query(query)
            return self._format_results(raw_results, search_query, method)
            
        except Exception as e:
            logger.error(f"Error in knowledge graph search: {e}")
            return f"Error searching knowledge graph: {str(e)}\n\nFalling back to basic search if available."

    async def ainvoke(self, query: str) -> str:
        """
        Async invoke method - wraps async search and formats results.
        This is primarily for Langchain tool invocation where string output is expected.
        """
        raw_results = await self._async_search(query)
        method, search_query = self._parse_query(query)
        return self._format_results(raw_results, search_query, method)

class KnowledgeGraphAnalysisTool(BaseTool):
    """
    Tool for analyzing knowledge graph structure and providing insights
    """
    
    name: str = "knowledge_graph_analysis"
    description: str = (
        "Analyze the knowledge graph structure and provide insights. "
        "Commands: 'stats' (general statistics), 'categories' (list main categories), "
        "'path:Category' (analyze specific category), 'similar:term' (find similar concepts)"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if EnhancedKnowledgeGraphTool._engine is None:
            _ = EnhancedKnowledgeGraphTool()

        self.engine = EnhancedKnowledgeGraphTool._engine
    
    def invoke(self, command: str) -> str:
        """
        Analyze knowledge graph based on command"""
        if not command.strip():
            return "Please provide an analysis command: 'stats', 'categories', 'path:Category', or 'similar:term'"
        
        try:
            if not self.engine:
                return "Knowledge graph engine not available."
            
            cmd = command.lower().strip()
            
            if cmd == "stats":
                return self.engine.query_graph("stats")
            elif cmd == "categories":
                return self.engine.query_graph("categories")
            elif cmd.startswith("path:"):
                category = command[5:].strip()
                return self.engine.query_graph(f"path:{category}")
            elif cmd.startswith("similar:"):
                term = command[8:].strip()
                relevant_docs = self.engine.retrieve_relevant_context(term)
                if relevant_docs:
                    return f"Similar concepts to '{term}':\n" + "\n".join([doc.page_content for doc in relevant_docs[:5]])
                return f"No similar concepts found for '{term}'."
            else:
                return "Unknown command. Use: 'stats', 'categories', 'path:Category', or 'similar:term'"
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return f"Analysis failed: {str(e)}"
    
def create_enhanced_kg_tools(kg_config: Dict[str, Any] = None) -> List[BaseTool]:
    """
    Factory function to create instances of enhanced knowledge graph tools.
    """
    return [
        EnhancedKnowledgeGraphTool(),
        KnowledgeGraphAnalysisTool()
    ] 