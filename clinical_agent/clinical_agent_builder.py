# --- clinical_agent/clinical_builder.py ---

import os
import logging
from typing import Dict, Any, Union, Optional, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --- Builder Imports (Same) ---
from rag_agent.rag_builder import RAGAgent
from web_search_agent.web_search_builder import WebSearchAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClinicalAgentBuilder")

# --- CLINICAL LOCAL STATE (NEW) ---
class ClinicalState(TypedDict):
    # Input/Context from Supervisor
    input: str
    context: Dict[str, Any]
    
    # RAG/Search/Reflection Results (Local Partition)
    clinical_query: str                  # The question to be answered
    rag_response: Dict[str, Any]         # Full result of the RAG step
    web_search_result: Dict[str, Any]    # Full result of the Web Search step
    reflection: Dict[str, Any]           # Decision of the Reflection node
    
    # Finalization
    final_candidate_answer: str          # The chosen text (RAG or Web Search) before final polish
    agent: str                           # Agent name for logging
    
# ... (rest of the ClinicalAgentBuilder class structure) ...

class ClinicalAgentBuilder:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        # ... (Initialization code, LLM setup, RAG/Web subgraph compilation) ...
        self.model = model
        self.temperature = temperature
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        # NOTE: Initialize LLM using OpenAI... (code omitted for brevity)
        self.llm = ChatOpenAI(model=self.model, temperature=self.temperature, openai_api_key=self.OPENAI_API_KEY)
        
        self.rag_agent = RAGAgent(index_name="your_clinical_index")
        self.web_agent = WebSearchAgent()
        self.RAG_SUBGRAPH = self.rag_agent.build()
        self.WEB_SUBGRAPH = self.web_agent.build()

    # --- Reflection Node (Logic adapted to read/write ClinicalState keys) ---
    def reflection_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Reads from state['rag_response'], writes to state['reflection']
        # ... (reflection logic) ...
        # (omitted for brevity, assume logic updated to use ClinicalState keys)
        state["reflection"] = {"satisfactory": True, "reason": "Placeholder"} # Example
        return state

    # --- LLM Polish Node (Logic adapted to read/write ClinicalState keys) ---
    def llm_polish_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Reads state['final_candidate_answer'], writes state['final_answer']
        # ... (polish logic) ...
        state["final_answer"] = "Placeholder polished answer." # Example
        return state
        
    # --- Final Output Node (NEW: To aggregate results for the parent graph) ---
    def clinical_output_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Maps the local clinical state back to the parent's overall state structure."""
        
        # 1. Map Final Output to Parent Key
        parent_update = {}
        parent_update["clinical_data"] = {
            "answer": state.get("final_answer"),
            "reflection": state.get("reflection"),
            "rag_response": state.get("rag_response"),
            "web_search_result": state.get("web_search_result"),
        }
        
        # 2. Update global UI output and history
        parent_update["agent"] = "CLINICAL_AGENT"
        parent_update["output"] = state.get("final_answer", "Clinical Error.")
        
        return parent_update


    def build(self):
        """Constructs and compiles the StateGraph."""
        logger.info("Building clinical agent graph: Integrating compiled subgraphs...")
        
        graph = StateGraph(ClinicalState) # Use the specific local schema

        # Add all nodes
        graph.add_node("rag", self.RAG_SUBGRAPH)
        graph.add_node("reflection", self.reflection_node)
        graph.add_node("web_search", self.WEB_SUBGRAPH)
        graph.add_node("polish", self.llm_polish_node)
        graph.add_node("output_map", self.clinical_output_node) # Final mapping node
        
        # ... (Edges remain the same, ensuring final edge points to output_map)
        graph.add_edge(START, "rag")
        # ... (rest of the conditional edges)
        graph.add_edge("polish", "output_map") # Polish -> Output Mapping
        graph.add_edge("output_map", END) # Output Mapping -> END
        
        return graph.compile()