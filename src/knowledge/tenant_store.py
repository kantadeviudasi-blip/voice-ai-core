import json
import os
from typing import Dict, Optional
from src.knowledge.prompt_builder import AgentProfile, PromptBuilder

class TenantStore:
    """
    Multi-tenant Agent Profile Store.
    Maintains customized voice agent profiles for 100+ businesses.
    """
    def __init__(self, storage_file: str = "tenants.json"):
        self.storage_file = storage_file
        self.profiles: Dict[str, AgentProfile] = {}
        self._init_defaults()

    def _init_defaults(self):
        # Default Template 1: Real Estate (Sunrise Heights)
        real_estate_text = """Sunrise Heights Jaipur. Luxury 2 BHK & 3 BHK flats at Jagatpura. 
Starting price 45 Lakhs. Zero brokerage, RERA approved, 80% loan assistance available. 
Amenities: Clubhouse, Swimming pool, 24/7 security, lush green gardens. 
Special Offer: Modular Kitchen free on bookings this weekend."""
        self.save_profile(PromptBuilder.synthesize_profile(
            tenant_id="real-estate-demo",
            company_name="Sunrise Heights",
            extracted_text=real_estate_text,
            agent_name="Sneha",
            language_mode="Hinglish",
            primary_goal="Check requirement (2BHK vs 3BHK) and book a free site visit for Saturday or Sunday"
        ))

        # Default Template 2: Healthcare Dental Clinic (Apex Dental)
        dental_text = """Apex Dental Care, Indiranagar Bangalore. 
Services: Teeth Cleaning, Root Canal Treatment, Invisible Aligners, Dental Implants. 
Senior Orthodontists with 15+ years experience. Painless treatment guarantee. 
Consultation fee: Only Rs. 299 for new patients this week."""
        self.save_profile(PromptBuilder.synthesize_profile(
            tenant_id="dental-clinic-demo",
            company_name="Apex Dental Care",
            extracted_text=dental_text,
            agent_name="Pooja",
            language_mode="Hinglish",
            primary_goal="Check dental issue and book doctor appointment slot"
        ))

        # Default Template 3: Solar Company (Apex Solar Solutions, Raipur)
        solar_text = """Apex Solar Solutions — Chhattisgarh ka No.1 Residential & Commercial Solar Provider.
Location: VIP Road, Raipur, Chhattisgarh. Contact: 9876543210. Website: www.apexsolarraipur.com.
Business Hours: Monday to Saturday, 9 AM to 7 PM.

ABOUT THE COMPANY:
Apex Solar Solutions Chhattisgarh ki leading residential aur commercial solar installation provider hai.
Hum rooftop solar power system installation, net-metering assistance, aur government subsidy processing mein specialize karte hain.

PRODUCTS & PRICING:
1. 3 kW On-Grid Solar System:
   - Price: Rs. 1,80,000 (before subsidy).
   - Government Subsidy: Up to Rs. 78,000 under PM Surya Ghar Muft Bijli Yojana.
   - Effective Price after Subsidy: Only Rs. 1,02,000.
   - Ideal for 2-3 BHK homes. Comfortably runs 2 ACs, Refrigerator, Fans, and all Lights.
   - Monthly electricity savings: Approx 350 to 400 units (Rs. 3,000 to Rs. 3,500 per month).

2. 5 kW On-Grid Solar System:
   - Price: Rs. 2,80,000 (before subsidy). Flat subsidy: Rs. 78,000.
   - Ideal for large homes and small offices.

KEY FEATURES & WARRANTIES:
- Solar Panel Warranty: 25 Years performance warranty.
- Inverter Warranty: 5 Years replacement warranty.
- Installation Time: Completed within 7 to 10 working days.
- Free Site Survey: 100% free site inspection before installation — no hidden charges.

PAYMENT & FINANCE OPTIONS:
- 0% Interest EMI available via Bajaj Finserv and HDFC Bank.
- Only 20% advance booking amount required. Remaining 80% only after installation and net-metering setup.

FAQS:
Q: Maintenance kitni hoti hai?
A: Bahut kam maintenance hoti hai. Panels ko sirf regular paani se 15 din mein ek baar saaf karna hota hai.

Q: Government subsidy kab milti hai?
A: Subsidy directly customer ke bank account mein credit hoti hai net-metering install hone ke 30 to 45 din ke andar.

LEAD QUALIFICATION QUESTIONS TO ASK:
- Aapka ghar kitne BHK ka hai?
- Monthly bijli bill kitna aata hai approximately?
- Kya aap Raipur mein hi hain?
- Key goal: Free site survey book karna — collect Name, Address, and preferred callback time slot."""

        apex_solar_profile = PromptBuilder.synthesize_profile(
            tenant_id="apex-solar-raipur",
            company_name="Apex Solar Solutions",
            extracted_text=solar_text,
            agent_name="Sneha",
            language_mode="Hinglish",
            primary_goal="Customer ka free site survey book karna. Pehle subsidy ka faayda batao, phir ghar ki details poocho (BHK, bijli bill), aur finally naam, address, aur callback time collect karo."
        )
        # Override the generic greeting with a conversion-optimised outbound hook
        apex_solar_profile.greeting = (
            "Namaste! Main Sneha bol rahi hoon, Apex Solar Solutions Raipur se. "
            "Sir, aapko pata hai government abhi 78 hazar rupaye ki subsidy de rahi hai solar lagwane par? "
            "Kya aap iske baare mein jaanna chahenge?"
        )
        self.save_profile(apex_solar_profile)

    def save_profile(self, profile: AgentProfile):
        self.profiles[profile.tenant_id] = profile

    def get_profile(self, tenant_id: str) -> Optional[AgentProfile]:
        return self.profiles.get(tenant_id) or self.profiles.get("real-estate-demo")

    def list_tenants(self) -> Dict[str, str]:
        return {tid: p.company_name for tid, p in self.profiles.items()}
