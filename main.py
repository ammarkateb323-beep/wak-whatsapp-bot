import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

import agent
import database
import memory
import whatsapp
from config import VERIFY_TOKEN, WEBHOOK_SECRET

# Basic logging setup so we can see what's happening in the terminal.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    App startup/shutdown lifecycle.
    Creates DB pool on startup and closes it on shutdown.
    """
    logger.info("Starting up - creating database connection pool...")
    await database.create_pool()
    logger.info("Database pool ready.")
    yield
    logger.info("Shutting down - closing database connection pool...")
    await database.close_pool()
    logger.info("Database pool closed.")


app = FastAPI(lifespan=lifespan)


@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    Meta webhook verification endpoint.
    """
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    logger.info("Webhook verification request received. Mode: %s", mode)

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("Webhook verified successfully.")
        return PlainTextResponse(content=challenge, status_code=200)

    logger.warning("Webhook verification failed. Token mismatch.")
    return PlainTextResponse(content="Forbidden", status_code=403)


@app.post("/webhook")
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    """
    Meta incoming-message webhook endpoint.
    Returns 200 quickly, then processes in background.
    """
    body = await request.json()
    logger.info("Webhook POST received.")

    try:
        # Meta payload structure:
        # body["entry"][0]["changes"][0]["value"]["messages"][0]
        entry = body.get("entry", [])
        changes = entry[0].get("changes", []) if entry else []
        value = changes[0].get("value", {}) if changes else {}

        messages_list = value.get("messages", [])
        if not messages_list:
            logger.info("No messages in payload - likely a status update. Ignoring.")
            return JSONResponse(content={"status": "ok"}, status_code=200)

        message = messages_list[0]
        if message.get("type") != "text":
            logger.info(
                "Non-text message received (type: %s). Ignoring.",
                message.get("type"),
            )
            return JSONResponse(content={"status": "ok"}, status_code=200)

        customer_phone = message.get("from")
        message_text = message.get("text", {}).get("body")

        if not customer_phone or not message_text:
            logger.warning("Missing phone number or message text. Ignoring.")
            return JSONResponse(content={"status": "ok"}, status_code=200)

        logger.info("Message from %s: %s", customer_phone, message_text[:50])

    except (IndexError, KeyError, TypeError) as exc:
        logger.error("Failed to parse webhook payload: %s", exc)
        # Always return 200 to prevent retry loops from Meta.
        return JSONResponse(content={"status": "ok"}, status_code=200)

    background_tasks.add_task(process_message, customer_phone, message_text)
    return JSONResponse(content={"status": "ok"}, status_code=200)


@app.post("/send")
async def send_agent_message(request: Request):
    """
    Called by the agent dashboard when the agent sends a reply.
    Validates the webhook secret, sends the message via WhatsApp,
    and saves it to the messages table with sender = 'agent'.
    """
    secret = request.headers.get("x-webhook-secret")
    if secret != WEBHOOK_SECRET:
        return JSONResponse(content={"error": "Forbidden"}, status_code=403)

    body = await request.json()
    customer_phone = body.get("customer_phone")
    message_text = body.get("message")

    if not customer_phone or not message_text:
        return JSONResponse(content={"error": "Missing fields"}, status_code=400)

    try:
        await whatsapp.send_message(to=customer_phone, text=message_text)
        await memory.save_message(
            customer_phone=customer_phone,
            direction="outbound",
            message_text=message_text,
            sender="agent",
        )
        return JSONResponse(content={"status": "sent"}, status_code=200)
    except Exception as exc:
        logger.error("Failed to send agent message: %s", exc)
        return JSONResponse(content={"error": str(exc)}, status_code=500)


async def process_message(customer_phone: str, message_text: str):
    """
    End-to-end processing in background:
    1) Generate reply via agent.py
    2) Send via WhatsApp Cloud API
    """
    try:
        logger.info("Processing message from %s...", customer_phone)

        reply = await agent.get_reply(
            customer_phone=customer_phone,
            new_message=message_text,
        )
        logger.info("Reply generated for %s: %s...", customer_phone, reply[:50])

        await whatsapp.send_message(to=customer_phone, text=reply)
        logger.info("Reply sent to %s successfully.", customer_phone)

    except Exception as exc:
        logger.error(
            "Failed to process message from %s: %s",
            customer_phone,
            exc,
            exc_info=True,
        )
