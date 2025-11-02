import os
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# --- BUILDER IMPORTS ---
from receptionist.receptionist_agent_builder import ReceptionistAgentBuilder 
from supervisor_agent.prompt import classification_prompt
from supervisor_agent.schemas import SupervisorOutput

# --- CONFIG ---
load_dotenv() 
SUPERVISOR_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required.")

# ----------------------------------------------------
# 1. STATE DEFINITION
# ----------------------------------------------------

class OverallSystemState(TypedDict):
    input: str
    history: List[Dict[str, str]]
    route: str
    agent: str
    patient_name: Optional[str]
    context: Dict[str, Any]
    receptionist_data: Dict[str, Any]
    clinical_data: Dict[str, Any]
    output: str

def create_initial_overall_state():
    return OverallSystemState(
        input="", 
        output="", 
        history=[], 
        agent="", 
        route="", 
        patient_name="",
        context={"patient_name": None}, 
        receptionist_data={}, 
        clinical_data={}
    )

# ----------------------------------------------------
# 2. SUPERVISOR AGENT BUILDER
# ----------------------------------------------------

class SupervisorAgentBuilder:
    def __init__(self):
        print("🔧 Initializing SupervisorAgentBuilder...")

        self.llm_supervisor = ChatOpenAI(
            model=SUPERVISOR_MODEL,
            temperature=0.0,
            openai_api_key=OPENAI_API_KEY
        )

        print("✅ Supervisor LLM initialized:", SUPERVISOR_MODEL)

        self.receptionist_builder = ReceptionistAgentBuilder()
        self.RECEPTIONIST_NODE = self.receptionist_builder.build()

        print("✅ Receptionist agent builder compiled successfully.\n")

    def supervisor_agent_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n🧠 [Supervisor Node Invoked]")
        print("Incoming state:", state)

        query = state["input"]
        updated_history = state["history"] + [{"role": "user", "content": query}]


        # --- Construct prompt ---
        CLASSIFICATION_PROMPT = """
        You are a routing supervisor. Your role is to determine the correct agent and attempt name extraction from the latest user query.
        ...
        LATEST Query to Analyze: {query}
        """

        history_str = "\n".join([f"{h['type'].capitalize()}: {h['content']}" for h in state["history"]])
        formatted_prompt = CLASSIFICATION_PROMPT.format(messages=history_str, query=query)

        structured_llm = self.llm_supervisor.with_structured_output(SupervisorOutput)

        print("📤 Sending prompt to LLM for routing decision...\n")
        result = structured_llm.invoke([HumanMessage(content=formatted_prompt)])

        print("📥 Received structured result:", result)

        route = result.route.strip().upper() if result.route else "RECEPTIONIST_AGENT"
        patient_name = result.patient_name

        print(f"🔀 Routing decision → {route}")
        print(f"👤 Extracted patient_name → {patient_name}\n")

        # Return partial update
        return {
            "input": query,
            "route": route,
            "history": updated_history,
            "patient_name": patient_name
        }

    def build(self):
        print("\nBuilding Master Graph...")
        graph = StateGraph(OverallSystemState)

        graph.add_node("SUPERVISOR_ROUTER", self.supervisor_agent_node)
        graph.add_node("RECEPTIONIST_AGENT", self.RECEPTIONIST_NODE)

        graph.add_edge(START, "SUPERVISOR_ROUTER")
        graph.add_edge("SUPERVISOR_ROUTER", "RECEPTIONIST_AGENT")
        graph.add_edge("RECEPTIONIST_AGENT", END)

        compiled = graph.compile()
        print("✅ Graph compiled successfully.\n")
        return compiled


# # ----------------------------------------------------
# # 3. TEST RUNNER (Run from terminal)
# # ----------------------------------------------------

# if __name__ == "__main__":
#     print("\n🚀 Starting test run of SupervisorAgentBuilder...\n")

#     supervisor = SupervisorAgentBuilder()
#     graph = supervisor.build()

#     # Create initial state
#     state = create_initial_overall_state()
#     state["input"] = "Hello, my name is Barbara Owens. I’d like to book an appointment."

#     print("\n▶️ Executing graph with initial query...\n")
#     result = graph.invoke(state)

#     print("\n================== FINAL OUTPUT ==================")
#     print(result)
#     print("==================================================\n")
