import os
import json
from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END, CompiledGraph
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# --- BUILDER IMPORTS (Must match your file structure) ---
# Assuming the file containing your builder classes is importable from the parent directory

# 1. CLINICAL AGENT IMPORTS
from clinical_agent.clinical_agent_builder import ClinicalAgentBuilder()
clinical_agent_builder = ClinicalAgentBuilder()
CLINICAL_GRAPH_COMPILED = clinical_agent_builder.build()

# 2. RECEPTIONIST AGENT IMPORTS
from receptionist.receptionist_agent_builder import ReceptionistAgentBuilder 
receptionist_agent_builder = ReceptionistAgentBuilder()
RECEPTIONIST_GRAPH_COMPILED = receptionist_agent_builder.build()

# Define the nodes that will hold the compiled graphs
RECEPTIONIST_NODE = RECEPTIONIST_GRAPH_COMPILED
CLINICAL_NODE = CLINICAL_GRAPH_COMPILED
   
from prompt import classification_prompt 

# --- CONFIGURATION ---
load_dotenv() 
model="gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required.")

# Initialize LLM for Supervisor Classification
llm_supervisor = ChatOpenAI(
    model=model,
    temperature=0.0,
    openai_api_key=OPENAI_API_KEY
)



def OverallState():
    # Includes standard LangGraph state variables and custom variables
    return {"input": "", "output": "", "history": [], "agent": "", "patient_name": None}





def supervisor_agent(state: Dict[str, Any]) -> str:
    """
    Uses an LLM to classify the query and returns the next agent's node name.
    """
    query = state["input"]
    print(f"\n--- Supervisor received query: '{query}' ---")

    formatted_prompt = classification_prompt.format(query=query)
    
    try:
        result = llm_supervisor.invoke([HumanMessage(content=formatted_prompt)])
        decision = result.content.strip().upper()
        
        if decision not in ["CLINICAL_AGENT", "RECEPTIONIST_AGENT"]:
            print(f"Classification failed: LLM returned '{decision}'. Defaulting to RECEPTIONIST_AGENT.")
            route = "RECEPTIONIST_AGENT"
        else:
            route = decision
            
    except Exception as e:
        print(f"ERROR: Supervisor LLM failed to invoke: {e}. Defaulting to RECEPTIONIST_AGENT.")
        route = "RECEPTIONIST_AGENT"

    print(f"Routing decision: → {route}")
    
    # The supervisor returns the string name of the next node
    return route




workflow = StateGraph(OverallState)

# 1. Add Nodes
# The nodes now hold the fully compiled graphs (runnables)
workflow.add_node("SUPERVISOR_ROUTER", supervisor_agent)
workflow.add_node("RECEPTIONIST_AGENT", RECEPTIONIST_NODE)
workflow.add_node("CLINICAL_AGENT", CLINICAL_NODE)

# 2. Define Routing Edges
workflow.add_edge(START, "SUPERVISOR_ROUTER")

# The conditional edge uses the output string returned by supervisor_agent to route
workflow.add_conditional_edges(
    "SUPERVISOR_ROUTER",
    lambda state: supervisor_agent(state), # Re-execute the supervisor logic to get the route key
    {
        "RECEPTIONIST_AGENT": "RECEPTIONIST_AGENT",
        "CLINICAL_AGENT": "CLINICAL_AGENT",
    }
)

# 3. Define End Edges (Agents complete the task and exit)
workflow.add_edge("RECEPTIONIST_AGENT", END)
workflow.add_edge("CLINICAL_AGENT", END)

# Compile the graph
app = workflow.compile()


if __name__ == "__main__":
    # --- Test Case 1: Simple/Greeting Query ---
    test_1 = {"input": "My name is John Smith. How are you today?", "patient_name": "John Smith", "history": []}
    print("--- Running Test 1: Simple Greeting ---")
    out_1 = app.invoke(test_1)
    print(f"\nFINAL AGENT: {out_1.get('agent')}")
    print(f"FINAL OUTPUT: {out_1.get('output')}")

    # --- Test Case 2: Clinical Query ---
    test_2 = {"input": "I have swelling in my legs. Should I take more medicine?", "patient_name": "John Smith", "history": []}
    print("\n--- Running Test 2: Clinical Question ---")
    out_2 = app.invoke(test_2)
    print(f"\nFINAL AGENT: {out_2.get('agent')}")
    print(f"FINAL OUTPUT: {out_2.get('output')}")
