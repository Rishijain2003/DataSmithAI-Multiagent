import json
import pandas as pd
import os
from pathlib import Path
import logging

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
PATIENT_DIR = BASE_DIR / "receptionist" / "dummy_patients"
PATIENT_INDEX = PATIENT_DIR / "patient_index.csv"

# Load patient index
patient_index = pd.read_csv(PATIENT_INDEX)


# from langchain_openai import ChatOpenAI
# model="gpt-4o-mini"
# temperature=0.7
# llm = ChatOpenAI(temperature=temperature, model_name=model)
from langchain_groq import ChatGroq

model = "llama-3.3-70b-versatile"  # or any other Groq-supported model
temperature = 0.7

llm = ChatGroq(
    temperature=temperature,
    model=model,
    # groq_api_key=""  # or set via environment variable
)

from prompt import receptionist_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_patient_file_by_name(name: str):
    """Look up patient's JSON file using the CSV index."""
    row = patient_index[patient_index['patient_name'].str.lower() == name.lower()]
    if row.empty:
        return None, None
    file_name = row.iloc[0]['file_name']
    file_path = PATIENT_DIR / file_name
    if not file_path.exists():
        return None, None
    with open(file_path, "r") as f:
        patient_data = json.load(f)
    return patient_data, file_name


def receptionist_agent(state):
    """
    Receptionist Agent:
    - Receives patient name and query from Supervisor Agent
    - Looks up JSON file for that patient
    - If found → passes JSON content to LLM
    - If not → returns "record not found"
    """
    query = state.get("input", "")
    patient_name = state.get("patient_name", "")

    print(f"Receptionist received: {query} for patient: {patient_name}")

    # --- Step 1: Fetch patient data
    patient_data, file_name = get_patient_file_by_name(patient_name)

    if not patient_data:
        response = f"Sorry, I couldn't find any discharge record for {patient_name}. Please verify the name."
    else:
        # --- Step 2: Pass JSON to LLM
        discharge_summary = json.dumps(patient_data, indent=2)
        prompt = f"""
        You are a hospital receptionist AI assistant. Use the patient’s discharge report below
        to greet them, summarize their condition, and ask how they are doing.

        Patient Discharge Report:
        {discharge_summary}

        Patient Query: {query}

        Respond politely and clearly, using information from the discharge summary when possible.
        """

        response = receptionist_prompt.format(discharge_summary=discharge_summary,query=query)
        
        #future I might orobably make it structured llm output
        #structured_llm = self.llm.with_structured_output(ReceptioinistOutput)
        result = llm.invoke(response)
        answer = result.content if hasattr(result, "content") else str(result)
        print(f"\n--- Receptionist LLM Response ---\n{answer}\n")


   
    state["agent"] = "Receptionist Agent"
    state["output"] = response
    state["history"].append({"role": "assistant", "content": response})
    state["query"]="Erik Keller"

   
    os.makedirs("./logs", exist_ok=True)
    with open("./logs/interactions.log", "a") as f:
        f.write(f"[Receptionist] {patient_name} | {query} → {response}\n")

    return state


if __name__ == "__main__":
    test_state = {
        "patient_name": "Erik Keller",
        "input": "Hey, how have I been recovering after my surgery?",
        "history": []
    }

    updated_state = receptionist_agent(test_state)
    print("\nFinal Agent Output:\n", updated_state["output"])
