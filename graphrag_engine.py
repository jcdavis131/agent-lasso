import os
import json
import logging
from neo4j import GraphDatabase
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain.chains import create_tagging_chain_pydantic, create_extraction_chain_pydantic
from pydantic import BaseModel, Field
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    MISTRAL = "mistral"

class NodeRelation(BaseModel):
    source: str = Field(..., description="Name of the source entity.")
    source_type: str = Field(..., description="Type of the source entity.")
    target: str = Field(..., description="Name of the target entity.")
    target_type: str = Field(..., description="Type of the target entity.")
    relation_type: str = Field(..., description="Type of the relationship between source and target entities.")
    description: Optional[str] = Field(None, description="Description of the relationship.")

class ExtractedGraphData(BaseModel):
    nodes: List[Dict[str, str]] = Field(..., description="List of extracted nodes with 'name' and 'type'.")
    relations: List[NodeRelation] = Field(..., description="List of extracted relationships.")

class BaseGraphRAG(ABC):
    """Abstract base class for GraphRAG implementations."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_for_extraction = self._initialize_llm(
            config.get("llm_for_extraction_provider"),
            config.get("llm_for_extraction_model")
        )

    @abstractmethod
    def ingest_document(self, document: Document):
        """Ingests a document into the knowledge graph."""
        pass

    @abstractmethod
    def query_graph(self, query: str) -> str:
        """Queries the knowledge graph."""
        pass

    @abstractmethod
    def retrieve_relevant_context(self, query: str) -> List[Document]:
        """Retrieves relevant context from the graph based on a query."""
        pass

    def _initialize_llm(self, provider: str, model_name: str):
        if provider == LLMProvider.OPENAI.value:
            return ChatOpenAI(model=model_name, temperature=0)
        elif provider == LLMProvider.ANTHROPIC.value:
            return ChatAnthropic(model=model_name, temperature=0)
        elif provider == LLMProvider.GROQ.value:
            return ChatGroq(model_name=model_name, temperature=0)
        elif provider == LLMProvider.MISTRAL.value:
            return ChatMistralAI(model=model_name, temperature=0)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _initialize_embedding_model(self, model_name: str):
        """Lazily create and return a `SentenceTransformer` instance.

        The heavy import is done inside the helper so the module global scope
        stays lightweight.
        """
        try:
            from importlib import import_module

            SentenceTransformer = import_module("sentence_transformers").SentenceTransformer
            return SentenceTransformer(model_name)
        except Exception as e:  # pragma: no cover
            logger.error(
                "Failed to load SentenceTransformer model %s: %s", model_name, e
            )
            raise

class Neo4jGraphRAG(BaseGraphRAG):
    """GraphRAG implementation using Neo4j as the knowledge graph."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.uri = config["neo4j_uri"]
        self.user = config["neo4j_user"]
        self.password = config["neo4j_password"]
        self.vector_index_name = config.get("vector_index_name", "embedding_index")
        
        embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.embedding_model = self._initialize_embedding_model(embedding_model_name)

        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()

        self.similarity_threshold = config.get("similarity_threshold", 0.75)
        self.enable_chunk_filtering = config.get("enable_chunk_filtering", True)
        self.graph = Neo4jGraph(uri=self.uri, username=self.user, password=self.password)
        self._initialize_vector_index()

    def _initialize_vector_index(self):
        try:
            self.graph.query(f"""
            CREATE VECTOR INDEX {self.vector_index_name} IF NOT EXISTS
            FOR (n:Chunk) ON (n.embedding) OPTIONS {{indexConfig: {{
                `vector.dimensions`: {self.embedding_dimension},
                `vector.similarity_function`: 'cosine'
            }}}}
            """)
            logger.info(f"Vector index {self.vector_index_name} ensured to exist.")
        except Exception as e:
            logger.error(f"Failed to create or ensure vector index {self.vector_index_name}: {e}")
            raise

    def ingest_document(self, document: Document):
        """
        Ingests a document into Neo4j.
        This involves:
        1. Extracting entities and relationships using an LLM.
        2. Creating nodes for entities if they don't exist.
        3. Creating relationships between entities.
        4. Storing document chunks with embeddings for RAG.
        """
        logger.info(f"Ingesting document: {document.metadata.get('source', 'Unknown Source')}")

        try:
            extraction_chain = create_extraction_chain_pydantic(ExtractedGraphData, self.llm_for_extraction)
            extracted_data = extraction_chain.invoke({"input": document.page_content})

            if not extracted_data or not extracted_data.get('text'):
                logger.warning(f"No graph data extracted from document content.")
                return

            parsed_data = ExtractedGraphData.parse_raw(extracted_data['text'])
            nodes_to_create = parsed_data.nodes
            relations_to_create = parsed_data.relations

            with self.graph.driver.session() as session:
                tx = session.begin_transaction()
                for node_data in nodes_to_create:
                    node_name = node_data.get("name")
                    node_type = node_data.get("type")
                    if node_name and node_type:
                        try:
                            tx.run(f"MERGE (n:`{node_type}` {{name: $name}})", name=node_name)
                            logger.debug(f"MERGED Node: {node_type} - {node_name}")
                        except Exception as e:
                            logger.error(f"Error merging node {node_name} ({node_type}): {e}")

                for rel_data in relations_to_create:
                    try:
                        query = (
                            f"MATCH (s:`{rel_data.source_type}` {{name: $source_name}}), "
                            f"(t:`{rel_data.target_type}` {{name: $target_name}}) "
                            f"MERGE (s)-[r:`{rel_data.relation_type}`]->(t) "
                            f"SET r.description = $description"
                        )
                        tx.run(query,
                                source_name=rel_data.source,
                                source_type=rel_data.source_type,
                                target_name=rel_data.target,
                                target_type=rel_data.target_type,
                                relation_type=rel_data.relation_type,
                                description=rel_data.description)
                        logger.debug(f"MERGED Relationship: {rel_data.source} - {rel_data.relation_type} -> {rel_data.target}")
                    except Exception as e:
                        logger.error(f"Error merging relationship {rel_data}: {e}")
                tx.commit()

            chunk_content = document.page_content
            chunk_embedding = self.embedding_model.encode(chunk_content) # Use .encode() for SentenceTransformer
            source_file = document.metadata.get('source', 'unknown')

            with self.graph.driver.session() as session:
                session.run(
                    f"CREATE (c:Chunk {{content: $content, source: $source, embedding: $embedding}})",
                    content=chunk_content,
                    source=source_file,
                    embedding=chunk_embedding.tolist() # Convert numpy array to list for Neo4j
                )
            logger.info(f"Stored document chunk from {source_file} in Neo4j.")

        except Exception as e:
            logger.error(f"Error during document ingestion: {e}")
            raise

    def query_graph(self, query: str) -> str:
        """
        Queries the Neo4j graph using a natural language query.
        This can involve:
        1. Converting NL query to Cypher (future enhancement, currently direct Cypher or semantic search).
        2. Executing Cypher.
        3. Retrieving context via vector search.
        4. Combining results.
        """
        try:
            relevant_chunks = self.retrieve_relevant_context(query)
            if relevant_chunks:
                context = "\n\n".join([doc.page_content for doc in relevant_chunks])
                return f"Retrieved relevant context:\n{context}"
            else:
                return "No relevant context found in the knowledge graph."
        except Exception as e:
            logger.error(f"Error querying graph: {e}")
            return f"Error querying graph: {e}"

    def retrieve_relevant_context(self, query: str) -> List[Document]:
        """
        Retrieves relevant document chunks from Neo4j using vector search.
        """
        try:
            vector_store = Neo4jVector(
                embedding=self.embedding_model.encode, # Pass the encode method directly for SentenceTransformer
                url=self.uri,
                username=self.user,
                password=self.password,
                index_name=self.vector_index_name,
                text_node_property="content",
                embedding_node_property="embedding",
            )
            results = vector_store.similarity_search_with_score(query, k=5)

            relevant_docs = []
            for doc, score in results:
                if score >= self.similarity_threshold:
                    relevant_docs.append(doc)
                    logger.debug(f"Retrieved relevant chunk (Score: {score}): {doc.page_content[:100]}...")
                else:
                    logger.debug(f"Chunk below similarity threshold (Score: {score}): {doc.page_content[:100]}...")
            return relevant_docs
        except Exception as e:
            logger.error(f"Error retrieving relevant context: {e}")
            return []

class JSONGraphRAG(BaseGraphRAG):
    """
    GraphRAG implementation using a simple JSON-based knowledge graph for fallback or testing.
    This is a simplified in-memory representation.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.graph_data = {"nodes": [], "relations": [], "chunks": []}
        self.enable_chunk_filtering = config.get("enable_chunk_filtering", False)

        embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.embedding_model = self._initialize_embedding_model(embedding_model_name)

    def ingest_document(self, document: Document):
        logger.info(f"Ingesting document into JSON Graph: {document.metadata.get('source', 'Unknown Source')}")
        nodes_extracted = [{"name": "SimulatedEntity1", "type": "Person"}]
        relations_extracted = [{"source": "SimulatedEntity1", "source_type": "Person", "target": "SimulatedEntity2", "target_type": "Org", "relation_type": "WORKS_FOR"}]

        self.graph_data["nodes"].extend(nodes_extracted)
        self.graph_data["relations"].extend(relations_extracted)

        chunk_content = document.page_content
        chunk_embedding = self.embedding_model.encode(chunk_content) # Use .encode() for SentenceTransformer
        source_file = document.metadata.get('source', 'unknown')

        self.graph_data["chunks"].append({
            "content": chunk_content,
            "source": source_file,
            "embedding": chunk_embedding.tolist()
        })
        logger.info(f"Document ingested into JSON graph (simulated).")

    def query_graph(self, query: str) -> str:
        relevant_chunks = [
            chunk["content"] for chunk in self.graph_data["chunks"]
            if query.lower() in chunk["content"].lower()
        ]
        if relevant_chunks:
            return "Found relevant information in JSON graph:\n" + "\n---\n".join(relevant_chunks)
        return "No relevant information found in JSON graph."

    def retrieve_relevant_context(self, query: str) -> List[Document]:
        results = []
        query_embedding = self.embedding_model.encode(query)

        for chunk_data in self.graph_data["chunks"]:
            chunk_embedding = np.array(chunk_data["embedding"])
            similarity = np.dot(query_embedding, chunk_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))

            if similarity >= self.config.get("similarity_threshold", 0.75):
                results.append(Document(page_content=chunk_data["content"], metadata={"source": chunk_data["source"]}))
        return results

class ChunkFilteringLLM:
    """
    Implements ChunkRAG by using an LLM to filter less relevant chunks.
    """
    def __init__(self, llm_for_filtering: Any):
        self.llm = llm_for_filtering
        class FilterDecision(BaseModel):
            is_relevant: bool = Field(..., description="True if the chunk is relevant to the query, False otherwise.")
            reason: Optional[str] = Field(None, description="Brief reason for the filtering decision.")

        self.filter_chain = create_tagging_chain_pydantic(FilterDecision, self.llm)

    def filter_chunks(self, query: str, chunks: List[Document]) -> List[Document]:
        if not chunks:
            return []

        filtered_chunks = []
        logger.info(f"Applying Chunk Filtering for query: '{query}' on {len(chunks)} chunks.")
        for i, chunk in enumerate(chunks):
            try:
                response = self.filter_chain.invoke({"input": f"Query: {query}\nChunk: {chunk.page_content}"})
                filter_decision = response.get('text')
                if filter_decision:
                    parsed_decision = FilterDecision.parse_raw(filter_decision)
                    if parsed_decision.is_relevant:
                        filtered_chunks.append(chunk)
                        logger.debug(f"Chunk {i+1} KEPT (Reason: {parsed_decision.reason or 'Relevant'}).")
                    else:
                        logger.debug(f"Chunk {i+1} FILTERED OUT (Reason: {parsed_decision.reason or 'Not relevant'}).")
                else:
                    logger.warning(f"LLM returned no filter decision for chunk {i+1}. Keeping by default.")
                    filtered_chunks.append(chunk)
            except Exception as e:
                logger.error(f"Error filtering chunk {i+1} with LLM: {e}. Keeping chunk by default.")
                filtered_chunks.append(chunk)
        logger.info(f"Filtered to {len(filtered_chunks)} relevant chunks.")
        return filtered_chunks

def migrate_json_to_neo4j(json_data: Dict[str, Any], neo4j_graph_rag: Neo4jGraphRAG):
    """
    Placeholder function for migrating existing JSON knowledge graph data to Neo4j.
    This would involve iterating through JSON entities/relationships and ingesting them.
    """
    logger.info("Starting JSON to Neo4j migration (placeholder).")
    for i, node in enumerate(json_data.get("nodes", [])):
        doc = Document(page_content=f"Node Name: {node.get('name')}, Type: {node.get('type')}",
                       metadata={"source": f"migrated_json_node_{i}"})
        try:
            neo4j_graph_rag.ingest_document(doc)
        except Exception as e:
            logger.error(f"Failed to migrate JSON node {node.get('name')}: {e}")

    for i, relation in enumerate(json_data.get("relations", [])):
        doc = Document(page_content=f"Relation from {relation.get('source')} to {relation.get('target')} of type {relation.get('relation_type')}",
                       metadata={"source": f"migrated_json_relation_{i}"})
        try:
            neo4j_graph_rag.ingest_document(doc)
        except Exception as e:
            logger.error(f"Failed to migrate JSON relation {relation}: {e}")

    logger.info("JSON to Neo4j migration completed (placeholder).")

def get_graph_rag_instance(config: Dict[str, Any]) -> BaseGraphRAG:
    """
    Factory function to get the appropriate GraphRAG instance based on configuration.
    """
    if config.get("enable_neo4j", False):
        try:
            return Neo4jGraphRAG(config)
        except Exception as e:
            logger.error(f"Failed to initialize Neo4jGraphRAG, falling back to JSONGraphRAG. Error: {e}")
            return JSONGraphRAG(config) # Fallback to JSON if Neo4j initialization fails
    else:
        return JSONGraphRAG(config)