"""
Refinement Agent for Daivis Labs System
Focuses on Grounded Hallucination Mitigation as per DO-RAG paper.
Cross-verifies LLM outputs against the knowledge graph and iteratively corrects inconsistencies.
"""

import asyncio
import logging
from typing import List, Dict, Any, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document # New import for Document

from config import get_llm, GRAPHRAG_CONFIG # Import GRAPHRAG_CONFIG
from graphrag_engine import get_graph_rag_instance, BaseGraphRAG # Updated import
from enhanced_kg_tool import EnhancedKnowledgeGraphTool # For direct KG queries if needed

logger = logging.getLogger(__name__)

class RefinementAgent:
    """
    Refinement agent to mitigate hallucinations by cross-verifying LLM outputs
    against the knowledge graph and iteratively correcting inconsistencies.
    """
    def __init__(self, llm_provider: str = "openai", model_name: str = "gpt-4o-mini"):
        self.llm = get_llm(llm_provider, model_name)
        self.kg_engine: BaseGraphRAG = get_graph_rag_instance(GRAPHRAG_CONFIG) # Initialize with factory function and config
        
        self.refinement_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                """You are a meticulous fact-checker and a highly accurate knowledge graph expert.
                Your goal is to ensure the generated answer is strictly grounded in the provided context and knowledge graph information.
                
                If the answer contains information NOT present in the context, identify it as a hallucination and suggest a corrected, grounded answer.
                If the answer is fully supported by the context, respond with "FULLY_GROUNDED".
                
                Context (from Knowledge Graph and Retrieval):
                {retrieved_context}
                
                Question: {question}
                Original Answer: {original_answer}
                
                Instructions:
                1. Analyze the 'Original Answer' against the 'Context'.
                2. If any part of the 'Original Answer' is not directly supported by the 'Context', identify it.
                3. If there are hallucinations, propose a 'Corrected Answer' that uses ONLY information from the 'Context'.
                4. If the 'Original Answer' is fully supported, output 'FULLY_GROUNDED'.
                
                Output Format (if hallucination detected):
                HALLUCINATION_DETECTED: [Brief description of hallucination]
                Corrected Answer: [Grounded and corrected answer]
                
                Output Format (if fully grounded):
                FULLY_GROUNDED
                """
            ),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        self.refinement_chain = (
            RunnablePassthrough.assign(
                retrieved_context=RunnableLambda(self._format_retrieved_context)
            ) | self.refinement_prompt | self.llm | StrOutputParser()
        )
    
    def _format_retrieved_context(self, state: Dict[str, Any]) -> str:
        """Formats the retrieved context for the prompt."""
        if 'retrieved_context' not in state or not state['retrieved_context']:
            return "No specific knowledge graph context was retrieved."
        
        formatted_context = []
        for doc in state['retrieved_context']:
            # Document object has page_content and metadata
            formatted_context.append(f"Source: {doc.metadata.get('source', 'Unknown Source')}")
            formatted_context.append(f"Content: {doc.page_content}")
            if doc.metadata:
                formatted_context.append(f"Metadata: {doc.metadata}")
            formatted_context.append("-" * 20)
            
        return "\n".join(formatted_context)
        
    async def refine_answer(self, state: Dict[str, Any]) -> str:
        """
        Refines the generated answer using the knowledge graph for hallucination mitigation.
        """
        if 'messages' not in state or not state['messages']:
            logger.warning("No messages in state for refinement.")
            return state.get('original_answer', '') # Return original if no messages
        
        original_answer_message = next((m for m in reversed(state['messages']) if isinstance(m, AIMessage)), None)
        if not original_answer_message:
            logger.warning("No AIMessage found for refinement.")
            return state.get('original_answer', '')

        original_answer = original_answer_message.content
        
        question_message = next((m for m in reversed(state['messages']) if isinstance(m, HumanMessage)), None)
        question = question_message.content if question_message else "No specific question provided."

        refinement_input = {
            "question": question,
            "original_answer": original_answer,
            "messages": state['messages'],
            "retrieved_context": state.get('retrieved_context', [])
        }
        
        try:
            refinement_output = await self.refinement_chain.ainvoke(refinement_input)
            
            if refinement_output.strip().upper() == "FULLY_GROUNDED":
                logger.info("Answer is fully grounded.")
                return original_answer
            elif refinement_output.startswith("HALLUCINATION_DETECTED:"):
                logger.warning(f"Hallucination detected: {refinement_output}")
                corrected_answer_line = next((line for line in refinement_output.split('\n') if line.startswith("Corrected Answer:")), None)
                if corrected_answer_line:
                    corrected_answer = corrected_answer_line.replace("Corrected Answer:", "").strip()
                    return corrected_answer
                else:
                    return f"Refinement suggested hallucination but no corrected answer found. Original: {original_answer}"
            else:
                logger.warning(f"Unexpected refinement output format: {refinement_output}")
                return original_answer
                
        except Exception as e:
            logger.error(f"Error during refinement: {e}")
            return original_answer 