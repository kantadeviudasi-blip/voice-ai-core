import httpx
from typing import Dict, Any, Optional

class WebhookNotifier:
    """
    Dispatches instant WhatsApp brochures, SMS reminders, and CRM webhooks after or during calls.
    """
    def __init__(self, crm_webhook_url: Optional[str] = None):
        self.crm_webhook_url = crm_webhook_url

    async def send_whatsapp_brochure(self, phone_number: str, company_name: str, document_url: str) -> Dict[str, Any]:
        """Simulates instant WhatsApp brochure dispatch."""
        return {
            "success": True,
            "channel": "WhatsApp",
            "recipient": phone_number,
            "message": f"Brochure from {company_name} dispatched to {phone_number} via WhatsApp."
        }

    async def trigger_crm_sync(self, payload: Dict[str, Any]) -> bool:
        if not self.crm_webhook_url:
            return True
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(self.crm_webhook_url, json=payload)
                return resp.status_code == 200
        except Exception:
            return False
