import httpx
from config import WHATSAPP_TOKEN, WHATSAPP_PHONE_ID


# The Meta WhatsApp Cloud API endpoint for sending messages.
# {WHATSAPP_PHONE_ID} identifies which WhatsApp number is sending.
# v19.0 is the stable API version - change only if Meta deprecates it.
WHATSAPP_API_URL = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_ID}/messages"

# The authorisation header sent with every request.
# Meta uses this Bearer token to confirm you have permission
# to send messages from this WhatsApp number.
HEADERS = {
    "Authorization": f"Bearer {WHATSAPP_TOKEN}",
    "Content-Type": "application/json",
}


async def send_message(to: str, text: str) -> None:
    """
    Sends a text message to a WhatsApp number via Meta's Cloud API.

    Called by main.py after agent.py returns the final reply.

    Args:
        to:   The customer's phone number in international format,
              no + or spaces. e.g. "971501234567" not "+971 50 123 4567"
              Meta's webhook gives us this format already via wa_id.
        text: The message text to send. Can include newlines and emoji.

    Returns:
        None. If the request fails, we raise an exception so main.py
        can log the error.
    """
    # The JSON body Meta expects for a simple text message.
    # "messaging_product" must always be "whatsapp" - it's required.
    # "type" tells Meta we're sending plain text, not an image or template.
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {
            "body": text,  # the actual message the customer will see
        },
    }

    # httpx.AsyncClient is the async HTTP client.
    # "async with" opens the client and closes it cleanly when done.
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=WHATSAPP_API_URL,
            headers=HEADERS,
            json=payload,  # httpx serialises the dict to JSON automatically
            timeout=10.0,  # fail after 10 seconds rather than hanging forever
        )

    # raise_for_status() checks if Meta returned a 4xx or 5xx error.
    # If something went wrong (bad token, invalid phone number, etc.)
    # this raises an httpx.HTTPStatusError with full response details.
    response.raise_for_status()
