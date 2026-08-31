import uuid
from typing import Dict, Any, List
from pydantic import BaseModel, Field

class Appointment(BaseModel):
    booking_id: str = Field(default_factory=lambda: f"APT-{uuid.uuid4().hex[:6].upper()}")
    customer_name: str
    phone_number: str
    date_str: str
    time_slot: str
    purpose: str = "Site Visit / Consultation"
    status: str = "Confirmed"

class AppointmentBooker:
    """
    Simulated Appointment & Calendar Booking Tool for Voice Agents.
    """
    def __init__(self):
        self.appointments: List[Appointment] = []

    def book_slot(
        self,
        customer_name: str,
        phone_number: str,
        date_str: str,
        time_slot: str,
        purpose: str = "Site Visit"
    ) -> Dict[str, Any]:
        appointment = Appointment(
            customer_name=customer_name,
            phone_number=phone_number,
            date_str=date_str,
            time_slot=time_slot,
            purpose=purpose
        )
        self.appointments.append(appointment)
        return {
            "success": True,
            "booking_id": appointment.booking_id,
            "details": f"Appointment booked for {customer_name} on {date_str} at {time_slot} ({purpose})."
        }
