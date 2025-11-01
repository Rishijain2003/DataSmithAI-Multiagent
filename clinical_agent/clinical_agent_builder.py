import os
import logging
from typing import Dict, Any, Union, Optional
from langgraph.graph import StateGraph, START, END, CompiledGraph
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import Any

# --- Builder Imports ---
# NOTE: These classes MUST be defined in your rag_builder.py and web_search_builder.py
try:
    from rag_builder import RAGAgent
    from web_search_builder import WebSearchAgent 
except ImportError as e:
    logging.error(f"Failed to import RAGAgent or WebSearchAgent: {e}. Cannot build graph.")
    # Define placeholders to prevent immediate crash, though functionality will fail.
    class RAGAgent:
        def __init__(self, index_name): pass
        def build(self): return lambda state: {"rag_result": {"answer": "RAG Builder Missing", "raw": {}}}
    class WebSearchAgent:
        def build(self): return lambda state: {"web_search_result": {"answer": "Web Builder Missing", "raw": {}}}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClinicalAgentBuilder")


# ----------------------------------------------------
# 1. CLINICAL AGENT BUILDER CLASS
# ----------------------------------------------------

class ClinicalAgentBuilder:
    """
    Constructs and manages the multi-step clinical agent workflow:
    RAG -> Reflection -> (Web Search) -> Polish -> END
    """
    def __init__(self, model: str = "llama-3.1-70b-versatile", temperature: float = 0.0):
        
        self.model = model
        self.temperature = temperature
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        
        # Initialize LLM for Reflection and Polish nodes
        if not self.GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set. Using hardcoded key for testing.")
            self.llm = ChatGroq(temperature=self.temperature, model=self.model, groq_api_key="gsk_...") 
        else:
            self.llm = ChatGroq(temperature=self.temperature, model=self.model, groq_api_key=self.GROQ_API_KEY)
            
        logger.info(f"Clinical Agent initialized with model: {self.model}")
        
        # Instantiate the builder classes (assuming RAGAgent requires an index name)
        self.rag_agent = RAGAgent(index_name="your_clinical_index")
        self.web_agent = WebSearchAgent()
        
        # Compile subgraphs once during initialization
        self.RAG_SUBGRAPH = self.rag_agent.build()
        self.WEB_SUBGRAPH = self.web_agent.build()


    # --------------------------
    # Node: Reflection Node
    # --------------------------
    def reflection_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LLM evaluates RAG answer sufficiency for conditional routing."""
        
        query = state.get("clinical_query", "")
        # Access the RAG result, which was written to state by the RAG subgraph
        rag_answer = state.get("rag_result", {}).get("answer", "")

        if not rag_answer or "Error:" in rag_answer or "Builder Missing" in rag_answer:
            state["reflection"] = {"satisfactory": False, "reason": "RAG returned no content or error."}
            return state

        reflection_prompt = (
            f"You are an expert clinical quality checker. Determine if the RAG Answer is **sufficient and relevant** "
            f"to address the patient's concern directly, or if it requires external web search. "
            f"Respond STRICTLY with a single word: **YES** (if satisfactory) or **NO** (if unsatisfactory)."
            f"\n\nPATIENT QUERY: \"{query}\"\n---RAG ANSWER: \"{rag_answer}\"\n\nDecision (YES/NO):"
        )
        
        messages = [
            SystemMessage(content="You are a strict clinical evaluation bot. Output only YES or NO."),
            HumanMessage(content=reflection_prompt)
        ]
        
        try:
            logger.info("Reflection node: LLM evaluating RAG answer sufficiency.")
            result = self.llm.invoke(messages)
            decision = result.content.strip().upper()
            is_satisfactory = decision == "YES"
            reason = f"LLM decided RAG answer is {'sufficient' if is_satisfactory else 'insufficient'}."
        except Exception as e:
            logger.exception(f"LLM Reflection failed: {e}. Defaulting to NOT satisfactory.")
            is_satisfactory = False
            reason = "LLM evaluation failed."

        state["reflection"] = {"satisfactory": is_satisfactory, "reason": reason}
        logger.info(f"Reflection decision: {state['reflection']['satisfactory']} ({reason})")
        return state

    
    # --------------------------
    # Node: LLM Polish / Aggregator Node
    # --------------------------
    def llm_polish_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Polishes the final candidate answer using the LLM."""
        
        # NOTE: This function must run after the final candidate has been selected 
        # (which is managed by the runner logic).
        candidate = state.get("final_candidate_answer", "")
        if not candidate:
            state["final_answer"] = ""
            return state

        prompt = (
            "You are a clinical assistant. Re-read the answer below and: "
            "1) produce a concise, polite, clinically-appropriate reply to the patient's question, "
            "2) include one-line summary of the evidence/source if available.\n\n"
            f"Answer to polish:\n{candidate}\n\nPolished concise reply:"
        )

        try:
            logger.info("LLM polish: invoking llm for final answer polishing.")
            result = self.llm.invoke(prompt)
            final_text = getattr(result, "content", str(result))
        except Exception as e:
            logger.exception("LLM polish failed, using candidate as final_answer.")
            final_text = candidate

        state["final_answer"] = final_text
        return state


    # --------------------------
    # Graph Builder
    # --------------------------
    def build(self) -> CompiledGraph:
        """Constructs and compiles the StateGraph."""
        logger.info("Building clinical agent graph: Integrating compiled subgraphs...")
        
        graph = StateGraph(schema=dict)

        # Add the compiled subgraphs and the reflection/polish nodes
        graph.add_node("rag", self.RAG_SUBGRAPH)
        graph.add_node("reflection", self.reflection_node)
        graph.add_node("web_search", self.WEB_SUBGRAPH)
        graph.add_node("polish", self.llm_polish_node)

        # Define the routing function
        def reflection_router(state: Dict[str, Any]) -> str:
            """Determines the next step based on the reflection result."""
            if state.get("reflection", {}).get("satisfactory", False):
                return "RAG_OK"
            else:
                return "NEEDS_SEARCH"

        # --- Define Edges ---
        graph.add_edge(START, "rag")
        graph.add_edge("rag", "reflection")
        
        # Conditional edge after reflection
        graph.add_conditional_edges(
            "reflection",
            reflection_router,
            {
                "RAG_OK": "polish",      # RAG is sufficient, skip web search
                "NEEDS_SEARCH": "web_search" # RAG failed, go to web search
            }
        )
        
        # Web search always proceeds to polish
        graph.add_edge("web_search", "polish")
        
        # The polish node is the final step
        graph.add_edge("polish", END)

        logger.info("Graph built and compiled successfully.")
        return graph.compile()


# --------------------------
# Graph Runner (For Testing/Invocation)
# --------------------------
def run_clinical_graph(initial_state: Dict[str, Any], graph: Any) -> Dict[str, Any]:
    """
    Executes the compiled LangGraph and performs final candidate selection.
    """
    logger.info(f"Clinical graph starting with query: {initial_state.get('clinical_query')}")
    
    # Invoke the graph
    final_state = graph.invoke(initial_state)

    # --- Manually set final_candidate_answer based on path taken ---
    # This logic determines which node's output (RAG or Web Search) should be polished
    satisfactory = final_state.get("reflection", {}).get("satisfactory", False)
    
    if satisfactory:
        # Candidate is the result from the RAG subgraph
        final_candidate_answer = final_state.get("rag_result", {}).get("answer", "")
    else:
        # Candidate is the result from the WEB SEARCH subgraph (if available)
        web_answer = final_state.get("web_search_result", {}).get("answer", "")
        if web_answer and "Error" not in web_answer and "Builder Missing" not in web_answer:
            final_candidate_answer = web_answer
        else:
            # Fallback to RAG result if web search failed
            final_candidate_answer = final_state.get("rag_result", {}).get("answer", "")
            
    # The polish node has already run using the final candidate answer, but we set the 
    # final_candidate_answer field here for inspection/logging integrity.
    final_state["final_candidate_answer"] = final_candidate_answer

    logger.info("Clinical graph run complete.")
    return final_state


# --------------------------
# Example standalone test (Requires successful builder imports)
# --------------------------
if __name__ == "__main__":
    # 1. Initialize the builder
    try:
        agent_builder = ClinicalAgentBuilder()
        graph_app = agent_builder.build()
    except Exception as e:
        logger.error(f"Initialization or Graph Build Failed: {e}")
        exit()

    # 2. Sample test state
    test_state = {
        "clinical_query": "I had a cholecystectomy two days ago. Is it normal to have mild fever and slight bile-colored drainage?",
        "history": []
    }

    try:
        # 3. Invoke the graph
        out_state = run_clinical_graph(test_state, graph_app)

        # 4. Print Results
        print("\n=== FINAL EXECUTION OUTPUT ===")
        print("Final Answer (Polished):\n", out_state.get("final_answer"))
        print("\nReflection Info:", out_state.get("reflection"))
        print("Candidate Source:", "RAG" if out_state.get("reflection", {}).get("satisfactory") else "WEB SEARCH (Fallback)")
        
    except Exception as e:
        logger.error(f"Graph execution failed during invoke: {e}")