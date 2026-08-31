from typing import Dict, Any, List
from pydantic import BaseModel, Field

class LeadInfo(BaseModel):
    caller_name: str = ""
    phone_number: str = ""
    requirement: str = ""
    budget_range: str = ""
    timeline: str = ""
    qualification_status: str = "In Progress"  # Qualified, Hot Lead, Warm, Cold, Disqualified
    notes: str = ""

class LeadQualifier:
    """
    Lead Qualification & CRM Logging Tool for Telecalling Agents.
    """
    def __init__(self):
        self.leads: List[LeadInfo] = []

    def qualify_lead(
        self,
        phone_number: str,
        caller_name: str = "",
        requirement: str = "",
        budget: str = "",
        timeline: str = "",
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Updates lead status and qualification score.
        """
        status = "Hot Lead" if ("lakh" in budget.lower() or "ready" in timeline.lower() or "urgent" in timeline.lower()) else "Qualified"
        
        lead = LeadInfo(
            caller_name=caller_name,
            phone_number=phone_number,
            requirement=requirement,
            budget_range=budget,
            timeline=timeline,
            qualification_status=status,
            notes=notes
        )
        self.leads.append(lead)
        return {
            "success": True,
            "status": status,
            "message": f"Lead for {phone_number} successfully recorded as {status}."
        }
