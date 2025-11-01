import os
import json
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Optional, TypedDict, List
# Changed LLM import to OpenAI
from langchain_openai import ChatOpenAI 
from langgraph.graph import StateGraph
from langgraph.graph import START, END
# Assuming prompt.py exists
from prompt import receptionist_prompt 
from dotenv import load_dotenv

# --- Configuration (Must match your file structure) ---
BASE_DIR = Path(__file__).resolve().parent.parent 
PATIENT_DIR = BASE_DIR / "receptionist" / "dummy_patients"
PATIENT_INDEX = PATIENT_DIR / "patient_index.csv"

# --- LLM Setup for OpenAI ---
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ReceptionistAgentBuilder")

# Load environment variables from .env file
load_dotenv() 



class ReceptionistState(TypedDict):
    """Defines the schema for the Receptionist Agent's state."""
    # Keys used throughout the agent's logic
    input: str
    patient_name: str
    output: str
    agent: str
    history: List[Dict[str, str]]

# --- Utility Functions (Unchanged) ---

def get_patient_file_by_name(name: str) -> tuple[Optional[Dict], Optional[str]]:
    """Look up patient's JSON file using the CSV index."""
    try:
        patient_index = pd.read_csv(PATIENT_INDEX)
    except FileNotFoundError:
        logger.error(f"Patient index file not found at: {PATIENT_INDEX}")
        return None, None
        
    row = patient_index[patient_index['patient_name'].str.lower() == name.lower()]
    if row.empty:
        return None, None
    
    file_name = row.iloc[0]['file_name']
    file_path = PATIENT_DIR / file_name
    
    if not file_path.exists():
        logger.error(f"Patient data file not found at: {file_path}")
        return None, None
        
    with open(file_path, "r") as f:
        patient_data = json.load(f)
        
    return patient_data, file_name



class ReceptionistAgentBuilder:
    """
    Builds the Receptionist Agent Node which handles patient record lookup 
    and initial LLM response generation, using ChatOpenAI.
    """
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE):
        
        # Initialize LLM using ChatOpenAI and expecting API key via environment variable
        self.llm = ChatOpenAI(
            temperature=temperature,
            model=model,
            openai_api_key=os.getenv("OPENAI_API_KEY") 
        )
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY is not set in environment.")
            
        logger.info(f"Receptionist Agent LLM initialized: {model}")

    
    def receptionist_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        The core node logic: Fetches patient data and generates LLM response.
        """
        query = state.get("input", "")
        patient_name = state.get("patient_name", "").strip()

        print(f"Receptionist received: '{query}' for patient: '{patient_name}'")

        # --- Step 1: Fetch patient data
        patient_data, file_name = get_patient_file_by_name(patient_name)
        response = ""

        if not patient_data:
            response = f"Sorry, I couldn't find any discharge record for {patient_name}. Please verify the name."
        else:
            # Case B: Record Found - Generate LLM Response
            discharge_summary = json.dumps(patient_data, indent=2)
            
            # Use the imported prompt template
            llm_input_prompt = receptionist_prompt.format(
                discharge_summary=discharge_summary,
                query=query
            )
            
            # --- Step 2: Pass prompt to LLM
            try:
                result = self.llm.invoke(llm_input_prompt)
                response = result.content if hasattr(result, "content") else str(result)
            except Exception as e:
                logger.error(f"LLM invocation error in ReceptionistAgent: {e}")
                response = "I apologize, but I am currently experiencing a system error. Please try again later."

        # --- Step 3: Update State and Log ---
        print(f"\n--- Receptionist LLM Response ---\n{response}\n")

        state["agent"] = "Receptionist Agent"
        state["output"] = response
        state["history"].append({"role": "assistant", "content": response})

        os.makedirs("./logs", exist_ok=True)
        with open("./logs/interactions.log", "a") as f:
            f.write(f"[Receptionist] {patient_name} | {query} -> {response}\n")

        return state


    def build(self):
        """
        Builds a simple, single-node LangGraph for the receptionist agent.
        """
        # FIX APPLIED: Pass the TypedDict class as the positional argument
        graph = StateGraph(ReceptionistState)

        # Add the core agent function as a node
        graph.add_node("receptionist_logic", self.receptionist_agent)

        # Define the simple workflow
        graph.add_edge(START, "receptionist_logic")
        graph.add_edge("receptionist_logic", END)
        
        return graph.compile()




if __name__ == "__main__":
    
    # 1. Initialize the builder
    receptionist_builder = ReceptionistAgentBuilder()
    
    # 2. Compile the graph
    receptionist_app = receptionist_builder.build()

    # 3. Test State (Requires dummy files and index to exist relative to BASE_DIR)
    test_state = {
        "patient_name": "Wanda Bennett",
        "input": "Hey, how have I been recovering after my surgery?",
        "history": []
    }

    print("\n--- Running Receptionist Agent ---")
    
    try:
        # Invoke the graph
        updated_state = receptionist_app.invoke(test_state)
        
        print("\n=== Final State Output ===")
        print(f"Agent Used: {updated_state['agent']}")
        print(f"Final Response:\n{updated_state['output']}")
        
    except FileNotFoundError as e:
        logger.error(f"\nFATAL: Necessary files missing. Ensure the path {PATIENT_INDEX} and patient files exist.")
        print(f"Error details: {e}")
    except Exception as e:
        logger.error(f"\nGraph execution failed: {e}")