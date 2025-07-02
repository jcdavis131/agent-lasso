"""
Daivis Labs Agent implementation using LangGraph.
Intelligence Meets Intuition.
Transform complex data into actionable insights through advanced AI.
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
import json
from datetime import datetime
import operator
import os
import difflib
import logging
import asyncio
import importlib

from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from tools import get_tools, get_tool_descriptions
from config import (
    MODEL_CONFIGS, 
    DEFAULT_MODEL_PROVIDER, 
    AGENT_CONFIG,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    MISTRAL_API_KEY,
    GROQ_API_KEY,
    GRAPHRAG_CONFIG,
    KNOWLEDGE_GRAPH_CONFIG,
    get_llm
)
from refinement_agent import RefinementAgent

# The knowledge-graph tools pull in very heavy dependencies (Neo4j, Sentence
# Transformers, etc.).  They are *not* required for the core landing-page
# workflow and would explode the Vercel bundle size.  We therefore import
# them lazily only if ENABLE_KG is set *and* the packages are installed.

ENABLE_KG = os.getenv("ENABLE_KG", "false").lower() == "true"

# Optional knowledge-graph tool placeholders (resolved at runtime if enabled)
if ENABLE_KG:
    try:
        kg_module = importlib.import_module("enhanced_kg_tool")
        EnhancedKnowledgeGraphTool = getattr(kg_module, "EnhancedKnowledgeGraphTool")
        KnowledgeGraphAnalysisTool = getattr(kg_module, "KnowledgeGraphAnalysisTool")
    except ModuleNotFoundError:  # Heavy deps not installed
        ENABLE_KG = False
        logging.warning(
            "ENABLE_KG was true but knowledge-graph dependencies are missing; "
            "KG tools will be disabled.  Add heavy deps and redeploy to enable."
        )

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """Represents the state of our agent."""
    messages: Annotated[List[BaseMessage], operator.add]
    tool_invocations: Annotated[List[Dict[str, Any]], operator.add]
    retrieved_context: Annotated[List[Document], operator.add]


# --- Knowledge Graph Tool Implementation ---
class KnowledgeGraphQueryTool(BaseTool):
    """
    Tool for querying the skills_data_clean.json knowledge graph.
    Supports fuzzy/partial matching and hierarchical (path-based) search.
    Optionally shows all subskills/children for a matched node.
    """
    name: str = "knowledge_graph_query"
    description: str = (
        "Query the skills knowledge graph for factual, structured information. "
        "Input: a natural language keyword, skill name, or path. "
        "Output: relevant skill descriptions and optionally subskills from the knowledge graph."
    )

    _kg_data = None
    _kg_path = os.path.join(os.path.dirname(__file__), "skills_data_clean.json")

    def __init__(self, kg_uri: str = None):
        super().__init__()
        if KnowledgeGraphQueryTool._kg_data is None:
            try:
                with open(self._kg_path, "r", encoding="utf-8") as f:
                    KnowledgeGraphQueryTool._kg_data = json.load(f)
            except Exception as e:
                KnowledgeGraphQueryTool._kg_data = {}
                print(f"[KG] Failed to load skills_data_clean.json: {e}")

    def invoke(self, query: str) -> str:
        """
        Perform fuzzy/partial and hierarchical search over the skills knowledge graph.
        Returns matching skill names, descriptions, and subskills if requested.
        """
        if not query.strip():
            return "Please provide a keyword, skill name, or path to search the knowledge graph."
        kg = KnowledgeGraphQueryTool._kg_data
        if not kg:
            return "Knowledge graph data is unavailable."
        query_lc = query.lower().strip()
        matches = []
        # If query starts with 'path:' treat as path-based search
        if query_lc.startswith('path:'):
            path = [p.strip() for p in query[5:].split('>') if p.strip()]
            node = kg
            for p in path:
                if isinstance(node, dict) and p in node:
                    node = node[p]
                else:
                    return f"No node found for path: {' > '.join(path)}"
            # Show all subskills/children
            def collect_children(n, prefix=None):
                if prefix is None:
                    prefix = []
                if isinstance(n, dict):
                    for k, v in n.items():
                        matches.append((" > ".join(prefix + [k]), v if isinstance(v, str) else "[Category]"))
                        collect_children(v, prefix + [k])
                elif isinstance(n, str):
                    matches.append((" > ".join(prefix), n))
            collect_children(node, path)
            if not matches:
                return f"No subskills found for path: {' > '.join(path)}"
            formatted = [f"🔎 **Subskills for path: {' > '.join(path)}**"]
            for i, (skill_path, desc) in enumerate(matches[:15], 1):
                formatted.append(f"{i}. **{skill_path}**\n   📝 {desc}")
            if len(matches) > 15:
                formatted.append(f"...and {len(matches) - 15} more results.")
            return "\n\n".join(formatted)
        # Otherwise, do fuzzy/partial match search
        def search_node(node, path=None):
            if path is None:
                path = []
            if isinstance(node, dict):
                for k, v in node.items():
                    # Fuzzy/partial match
                    if query_lc in k.lower() or difflib.get_close_matches(query_lc, [k.lower()], n=1, cutoff=0.7):
                        matches.append((" > ".join(path + [k]), v if isinstance(v, str) else "[Category]"))
                    search_node(v, path + [k])
            elif isinstance(node, str):
                if query_lc in node.lower() or difflib.get_close_matches(query_lc, [node.lower()], n=1, cutoff=0.7):
                    matches.append((" > ".join(path), node))
        search_node(kg)
        if not matches:
            return f"No results found for '{query}' in the skills knowledge graph.\n\nTip: To browse a category, use 'path:Category > Subcategory' syntax."
        formatted = [f"🔎 **Results for '{query}':**"]
        for i, (skill_path, desc) in enumerate(matches[:15], 1):
            formatted.append(f"{i}. **{skill_path}**\n   📝 {desc}")
        if len(matches) > 15:
            formatted.append(f"...and {len(matches) - 15} more results.")
        formatted.append("\n💡 Tip: To browse a category, use 'path:Category > Subcategory' (e.g., path:Information Technology > Microsoft Development Tools)")
        return "\n\n".join(formatted)


class DaivisAgent:
    """
    Daivis Labs Agent powered by LangGraph.
    Now supports optional knowledge graph integration as a tool.
    """
    def __init__(self, model_provider: str, model_name: str, tools_config: List[str], system_prompt: str = None, refinement_agent: Optional[RefinementAgent] = None, custom_api_keys: dict = None):
        self.model_provider = model_provider
        self.model_name = model_name
        self.custom_api_keys = custom_api_keys
        self.refinement_agent = refinement_agent  # Assign early so it's available in _create_enhanced_system_prompt
        
        # Get all available tools and filter based on tools_config
        all_tools = get_tools()
        self.tools = [tool for tool in all_tools if tool.name in tools_config]
        self.tool_map = {tool.name: tool for tool in self.tools}

        self.llm = get_llm(model_provider, model_name, custom_api_keys=custom_api_keys)
        if not system_prompt:
            system_prompt = self._create_enhanced_system_prompt()
        self.system_prompt = system_prompt

        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()
        self.session_states = {}
        self.memory = None
        self.verbose = False
        self.recursion_limit = 50
        self.app = self.graph

    def _create_enhanced_system_prompt(self) -> str:
        """Create an enhanced system prompt that encourages tool usage, including KG if present."""
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(f"- `{tool.name}`: {tool.description}")
        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "No tools available"

        kg_instructions = ""
        # Check if a refinement agent with KG capabilities is present
        if self.refinement_agent and hasattr(self.refinement_agent, 'kg_engine'):
            kg_instructions = (
                "\n7. **For knowledge and skills queries** - PRIORITIZE using external knowledge retrieval and refinement. Do NOT directly query knowledge graph tools unless explicitly instructed and the tool is available. If you believe information is in the knowledge graph, prompt for retrieval and refinement.\n"
            )
        
        return f"""You are an elite AI specialist from Daivis Labs—a fusion of analytical prowess and creative insight.

Your mission: Transform complexity into clarity, uncertainty into understanding.

Your expertise transforms raw data into transformative insights that drive meaningful outcomes.

**Available Tools:**
{tools_text}

**Tool Usage Guidelines:**
1. **ALWAYS use tools** when appropriate - don't try to answer from memory if a tool can provide better information
2. **For math questions** - ALWAYS use the calculator tool, even for simple calculations
3. **For time/date questions** - ALWAYS use the current_time tool
4. **For research/current events** - ALWAYS use search tools
5. **For weather** - ALWAYS use the weather tool
6. **For file operations** - ALWAYS use file_operations tool
{kg_instructions}
**Key Rules:**
- If user asks "what time is it?" → MUST use current_time tool
- If user asks any math question → MUST use calculator tool  
- If user asks about current events → MUST use search tools
- If user asks about weather → MUST use weather tool
- If user provides data to analyze → MUST use data_analysis tool
- Always explain which tools you're using and why
- Be proactive with tool usage - it's better to use a tool than give incomplete answers
"""

    def _build_graph(self):
        """Builds the LangGraph computational graph."""
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_model)
        graph.add_node("tools", self.tool_node)
        graph.add_conditional_edges("llm", self.should_continue)
        graph.add_edge("tools", "llm")
        graph.set_entry_point("llm")
        return graph.compile()

    def get_session_state(self, session_id: str) -> AgentState:
        """Retrieves or initializes the session state."""
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                "messages": [],
                "tool_invocations": [],
                "retrieved_context": []
            }
        return self.session_states[session_id]

    def should_continue(self, state: AgentState):
        """Determines whether the agent should continue or end."""
        last_message = state['messages'][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "__end__"

    def call_model(self, state: AgentState):
        """Calls the language model and adds its response to the state."""
        messages = state['messages']
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def tool_node(self, state: AgentState):
        """Executes tool calls and adds results to the state."""
        last_message = state['messages'][-1]
        tool_invocations = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_function = self.tool_map.get(tool_name)
            if tool_function:
                try:
                    output = tool_function.invoke(tool_args)
                    tool_invocations.append({
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "output": output
                    })
                    state['messages'].append(ToolMessage(content=str(output), tool_call_id=tool_call['id']))
                except Exception as e:
                    error_msg = f"Error calling tool {tool_name}: {e}"
                    logger.error(error_msg)
                    tool_invocations.append({
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "error": error_msg
                    })
                    state['messages'].append(ToolMessage(content=error_msg, tool_call_id=tool_call['id']))
            else:
                error_msg = f"Tool {tool_name} not found."
                logger.warning(error_msg)
                tool_invocations.append({
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "error": error_msg
                })
                state['messages'].append(ToolMessage(content=error_msg, tool_call_id=tool_call['id']))
        return {"tool_invocations": tool_invocations}

    async def arun(self, query: str, session_id: str) -> Dict[str, Any]:
        """Asynchronously runs the agent with a given query and session ID."""
        config = {"configurable": {"thread_id": session_id}}
        inputs = {"messages": [HumanMessage(content=query)]}
        
        async for event in self.app.astream(inputs, config=config, recursion_limit=self.recursion_limit):
            for key, value in event.items():
                if key == "llm" and "messages" in value:
                    last_llm_message = value["messages"][-1]
                    if isinstance(last_llm_message, AIMessage) and not last_llm_message.tool_calls:
                        
                        # Retrieve context and attempt refinement if refinement_agent is present
                        retrieved_context = self.session_states[session_id].get('retrieved_context', [])
                        original_answer = last_llm_message.content

                        if self.refinement_agent:
                            logger.info(f"Attempting refinement for session {session_id}...")
                            refinement_state = {
                                "messages": self.session_states[session_id]["messages"],
                                "original_answer": original_answer,
                                "retrieved_context": retrieved_context # Pass retrieved context
                            }
                            try:
                                refined_answer = await self.refinement_agent.refine_answer(refinement_state)
                                if refined_answer != original_answer:
                                    logger.info(f"Answer refined successfully for session {session_id}.")
                                    # Update the last message with the refined answer
                                    self.session_states[session_id]["messages"][-1].content = refined_answer
                                else:
                                    logger.info(f"Answer fully grounded by refinement for session {session_id}.")
                                return {"answer": refined_answer}
                            except Exception as e:
                                logger.error(f"Refinement agent failed for session {session_id}: {e}")
                                return {"answer": original_answer} # Fallback to original
                        else:
                            logger.info(f"No refinement agent configured for session {session_id}.")
                            return {"answer": original_answer}

        final_state = self.app.get_state(config)
        final_messages = final_state.values["messages"]
        final_answer = final_messages[-1].content if final_messages else "No response."
        return {"answer": final_answer}

    def run(
        self, 
        message: str, 
        thread_id: str = "default",
        stream: bool = False
    ) -> str:
        """
        Synchronously runs the agent with a given message and thread ID.
        """
        if stream:
            raise NotImplementedError("Streaming not implemented for synchronous run.")
        
        # For synchronous calls, we will simulate the async behavior.
        # This is a simplified approach; a real-world scenario might use asyncio.run(self.arun(...))
        # if this method needs to be genuinely synchronous but built on async logic.
        session_id = thread_id
        self.session_states[session_id] = {
            "messages": [HumanMessage(content=message)],
            "tool_invocations": [],
            "retrieved_context": []
        }

        # Simulate graph execution steps
        # Call LLM
        llm_output = self.call_model(self.session_states[session_id])
        self.session_states[session_id]["messages"].extend(llm_output["messages"])

        # Check if should continue (i.e., tool call)
        if self.should_continue(self.session_states[session_id]) == "tools":
            tool_output = self.tool_node(self.session_states[session_id])
            self.session_states[session_id]["tool_invocations"].extend(tool_output["tool_invocations"])
            
            # After tool call, call LLM again to interpret tool output
            llm_output = self.call_model(self.session_states[session_id])
            self.session_states[session_id]["messages"].extend(llm_output["messages"])

        # Apply refinement if enabled
        original_answer = self.session_states[session_id]["messages"][-1].content
        if self.refinement_agent:
            logger.info(f"Attempting synchronous refinement for session {session_id}...")
            refinement_state = {
                "messages": self.session_states[session_id]["messages"],
                "original_answer": original_answer,
                "retrieved_context": self.session_states[session_id].get('retrieved_context', [])
            }
            try:
                refined_answer = asyncio.run(self.refinement_agent.refine_answer(refinement_state))
                if refined_answer != original_answer:
                    logger.info(f"Answer refined successfully for session {session_id}.")
                    self.session_states[session_id]["messages"][-1].content = refined_answer
                    return refined_answer
                else:
                    logger.info(f"Answer fully grounded by refinement for session {session_id}.")
                    return original_answer
            except Exception as e:
                logger.error(f"Synchronous refinement agent failed for session {session_id}: {e}")
                return original_answer
        else:
            logger.info(f"No refinement agent configured for session {session_id}.")
            return original_answer

    def get_conversation_history(self, thread_id: str = "default") -> List[Dict[str, str]]:
        """Retrieves the conversation history for a given thread ID."""
        state = self.get_session_state(thread_id)
        history = []
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "agent", "content": msg.content})
        return history

    def clear_memory(self, thread_id: str = "default"):
        """Clears the conversation memory for a given thread ID."""
        if thread_id in self.session_states:
            del self.session_states[thread_id]

    def get_available_tools(self) -> List[str]:
        """Returns a list of names of the tools available to the agent."""
        return [tool.name for tool in self.tools]

    def get_model_info(self) -> Dict[str, str]:
        """Returns information about the language model used by the agent."""
        return {
            "provider": self.model_provider,
            "name": self.model_name
        }



# Helper to dynamically load tools based on configuration
# This would ideally be in a separate tools.py module
# (currently located there, but this is a reminder)
# def get_tools():
#     return [
#         duckduckgo_search, 
#         calculator,
# #        CurrentTimeTool(),
# #        WeatherTool() # Add weather tool
#     ]

# NOTE: For deployment, ensure environment variables are set:
# OPENAI_API_KEY, ANTHROPIC_API_KEY, MISTRAL_API_KEY, GROQ_API_KEY
