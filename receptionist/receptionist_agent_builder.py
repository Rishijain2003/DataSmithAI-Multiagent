import os
import json
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Optional, TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
PATIENT_DIR = BASE_DIR / "receptionist" / "dummy_patients"
PATIENT_INDEX = PATIENT_DIR / "patient_index.csv"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7

# --- Prompt Template ---
receptionist_prompt = """
You are a hospital receptionist AI assistant. Use the patient’s discharge report below
to greet them, summarize their condition, and ask how they are doing.

Patient Name: {patient_name}

Patient Discharge Report:
{discharge_summary}

Conversation History:
{messages}

Patient Query: {query}

Respond politely and clearly, using information from the discharge summary when possible.
If the patient's message is just their name or a greeting, respond with a short summary
of their discharge report and ask how they are feeling today.
"""

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ReceptionistAgentBuilder")
load_dotenv()


# ----------------------------------------------------
# RECEPTIONIST LOCAL STATE
# ----------------------------------------------------
class ReceptionistState(TypedDict):
    input: str
    history: List[Dict[str, str]]
    patient_name: str

    # Local Output Partition Key
    discharge_summary: str
    response_text: str
    agent: str


# --- Utility: Patient lookup ---
def get_patient_file_by_name(name: str) -> tuple[Optional[Dict], Optional[str]]:
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


# ----------------------------------------------------
# RECEPTIONIST AGENT BUILDER
# ----------------------------------------------------
class ReceptionistAgentBuilder:
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE):
        self.llm = ChatOpenAI(
            temperature=temperature,
            model=model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY is not set in environment.")
        logger.info(f"Receptionist Agent LLM initialized: {model}")

    def receptionist_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core receptionist logic — reads patient info, retrieves discharge data,
        and crafts a friendly contextual response entirely via the LLM.
        """
        query = state.get("input", "")
        history = state.get("history", [])
        patient_name = state.get("patient_name", "").strip()

        print(f"[Receptionist] Received: '{query}' | Patient: '{patient_name}'")

        patient_data, _ = get_patient_file_by_name(patient_name)
        if not patient_data:
            response_text = f"Sorry, I couldn't find any discharge record for {patient_name}. Please verify the name."
            discharge_summary = ""
        else:
            discharge_summary = json.dumps(patient_data, indent=2)

            # Format conversation history
            messages_str = "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in history])

            # Build prompt
            llm_input_prompt = receptionist_prompt.format(
                patient_name=patient_name,
                discharge_summary=discharge_summary,
                query=query,
                messages=messages_str,
            )

            try:
                result = self.llm.invoke(llm_input_prompt)
                response_text = result.content if hasattr(result, "content") else str(result)
            except Exception as e:
                logger.error(f"LLM invocation error in ReceptionistAgent: {e}")
                response_text = (
                    "I apologize, but I am currently experiencing a system error. Please try again later."
                )

        # --- Write back to state ---
        state["discharge_summary"] = discharge_summary
        state["response_text"] = response_text
        state["agent"] = "Receptionist Agent"
        state["output"] = response_text
        state["history"].append({"role": "assistant", "content": response_text})

        os.makedirs("./logs", exist_ok=True)
        with open("./logs/interactions.log", "a") as f:
            f.write(f"[Receptionist] {patient_name} | {query} -> {response_text}\n")

        return state

    def build(self):
        """Builds a simple single-node LangGraph for the receptionist agent."""
        graph = StateGraph(ReceptionistState)
        graph.add_node("receptionist_logic", self.receptionist_agent)
        graph.add_edge(START, "receptionist_logic")
        graph.add_edge("receptionist_logic", END)
        return graph.compile()
