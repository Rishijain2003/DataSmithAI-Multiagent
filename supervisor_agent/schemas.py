from pydantic import BaseModel, Field
from typing import Optional

class SupervisorOutput(BaseModel):
    """Pydantic model for the supervisor's routing decision and context extraction."""
    route: str = Field(
        ...,
        description="The name of the agent to handle the user's query. Must be 'CLINICAL_AGENT' or 'RECEPTIONIST_AGENT'."
    )
    patient_name: Optional[str] = Field(
        None,
        description="If the user's latest query or history contains a name and the 'patient_name' is currently unknown (None), extract the full name. Otherwise, return None."
    )