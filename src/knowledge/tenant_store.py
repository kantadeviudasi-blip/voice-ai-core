import json
import os
from typing import Dict, Optional
from src.knowledge.prompt_builder import AgentProfile, PromptBuilder

class TenantStore:
    """
    Multi-tenant Agent Profile Store.
    Maintains customized voice agent profiles for Rooftop Solar & PM Surya Ghar Yojana.
    """
    def __init__(self, storage_file: str = "tenants.json"):
        self.storage_file = storage_file
        self.profiles: Dict[str, AgentProfile] = {}
        self._init_defaults()

    def _init_defaults(self):
        # 1. Primary Profile: Apex Solar Solutions (PM Surya Ghar Muft Bijli Yojana)
        solar_text_apex = """Apex Solar Solutions — PM Surya Ghar Muft Bijli Yojana Authorized Channel Partner.
Location: Raipur, Chhattisgarh. Contact: 9876543210.
Business Hours: Monday to Saturday, 9 AM to 7 PM.

ABOUT THE SERVICE:
Apex Solar Solutions chhaton (rooftop) par solar panel lagane aur PM Surya Ghar Yojana ke tehat government subsidy dilwane ka kaam karti hai.
Hum free site survey, net-metering setup, aur 0% EMI finance provide karte hain.

PRODUCTS & PRICING:
1. 3 kW On-Grid Rooftop Solar System:
   - Kul Kharcha (Before Subsidy): Rs. 1,80,000.
   - Government Subsidy (PM Surya Ghar): Flat Rs. 78,000 direct bank account mein.
   - Effective Price after Subsidy: Only Rs. 1,02,000.
   - Ideal for 2 to 4 room houses. Runs 2 ACs, Refrigerator, Washing Machine, Fans, and Lights.
   - Monthly electricity savings: Approx 350 to 400 units (Rs. 3,000 to Rs. 4,000 per month savings). Bill almost ZERO ho jaata hai.

2. 5 kW On-Grid Solar System:
   - Total Price: Rs. 2,80,000 (before subsidy).
   - Government Subsidy: Flat Rs. 78,000.
   - Ideal for large homes, bungalows, and small shops.

WARRANTY & INSTALLATION:
- Solar Panels Warranty: 25 Years Performance Warranty.
- Inverter Warranty: 5 Years Replacement Warranty.
- Installation Time: 7 to 10 working days.
- Free Site Feasibility Survey: 100% Free technical rooftop survey across city.

FINANCE & EMI:
- 0% Interest EMI available (Bajaj Finserv & HDFC Bank).
- Monthly EMI starting at only Rs. 2,500/month. Jo bijli ka bill bachta hai, usi se EMI nikal aati hai.

ROOFTOP SPACE REQUIRED:
- 3 kW solar ke liye lagbhag 300 square feet chhat ki jagah chahiye hoti hai jahan dhoop aati ho.
- 5 kW ke liye lagbhag 500 square feet jagah chahiye.

FAQS:
Q: Subsidy kaise milti hai?
A: PM Surya Ghar portal par registration aur net-metering lagne ke 30 din ke andar 78,000 rupaye sidhe customer ke bank account mein aate hain.

Q: Maintenance kitna hota hai?
A: Koi khaas maintenance nahi, bas 15 din mein ek baar sadharan paani se panels ko dho lijiye.

LEAD QUALIFICATION FLOW:
1. Check bijli bill: Monthly bill lagbhag kitna aata hai?
2. Check roof space: Ghar ki chhat apni hai ya rented?
3. Book Free Site Survey: Customer ka naam, address aur survey ka time confirm karna."""

        apex_profile = PromptBuilder.synthesize_profile(
            tenant_id="apex-solar-solutions",
            company_name="Apex Solar Solutions",
            extracted_text=solar_text_apex,
            agent_name="स्नेहा",
            language_mode="Hindi",
            primary_goal="ग्राहक को पीएम सूर्य घर योजना की 78,000 रुपये सब्सिडी समझाना, मासिक बिजली बिल पूछना और फ्री साइट सर्वे बुक करना।"
        )
        apex_profile.greeting = "नमस्ते सर! मैं अपेक्स सोलर से स्नेहा बोल रही हूँ। सर, पीएम सूर्य घर योजना के तहत 78,000 रुपये तक की सरकारी सब्सिडी मिल रही है। क्या आपका बिजली बिल ज्यादा आता है सर?"
        self.save_profile(apex_profile)

        # 2. Secondary Profile: GreenTech Solar Solutions
        solar_text_greentech = """GreenTech Solar Solutions India.
Authorized Solar EPC installer under PM Surya Ghar Muft Bijli Yojana.
Offering 3kW and 5kW Tier-1 Mono PERC Solar Panels with 25 Years Warranty.
Zero Investment Rooftop Solar with Easy EMI and Rs. 78,000 Direct DBT Subsidy."""
        
        greentech_profile = PromptBuilder.synthesize_profile(
            tenant_id="greentech-solar",
            company_name="GreenTech Solar",
            extracted_text=solar_text_greentech,
            agent_name="स्नेहा",
            language_mode="Hindi",
            primary_goal="पीएम सूर्य घर मुफ्त बिजली योजना के तहत 78,000 सब्सिडी की जानकारी देना और फ्री रूफटॉप इंस्पेक्शन शेड्यूल करना।"
        )
        greentech_profile.greeting = "नमस्ते सर! मैं ग्रीनटेक सोलर से स्नेहा बात कर रही हूँ। क्या आप अपनी छत पर सोलर लगवाकर बिजली बिल जीरो करना चाहते हैं सर?"
        self.save_profile(greentech_profile)

    def save_profile(self, profile: AgentProfile):
        self.profiles[profile.tenant_id] = profile

    def get_profile(self, tenant_id: str) -> Optional[AgentProfile]:
        return self.profiles.get(tenant_id) or self.profiles.get("apex-solar-solutions")

    def list_tenants(self) -> Dict[str, str]:
        return {tid: p.company_name for tid, p in self.profiles.items()}

