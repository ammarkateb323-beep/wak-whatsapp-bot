import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

import agent
import database
import memory
import transcribe as transcribe_mod
import whatsapp
from config import VERIFY_TOKEN, WEBHOOK_SECRET
from notifications import mask_phone

# ---------------------------------------------------------------------------
# Structured logging — format: timestamp | LEVEL | module | message
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background job: send Jitsi links 15 minutes before meetings
# ---------------------------------------------------------------------------


async def _link_delivery_loop():
    """
    Background task that runs every 60 seconds.
    Sends meeting links to customers whose meetings start within 15 minutes.
    """
    while True:
        await asyncio.sleep(60)
        try:
            meetings = await database.get_meetings_to_notify()
            for m in meetings:
                raw_base = (os.environ.get("APP_URL") or "wak-agents.up.railway.app").rstrip("/")
                base_url = raw_base if raw_base.startswith("http") else f"https://{raw_base}"
                meeting_url = (
                    f"{base_url}/meeting/{m['meeting_token']}"
                    if m.get("meeting_token")
                    else m["meeting_link"]
                )
                msg = f"Your meeting is starting soon! Join here: {meeting_url}"
                await whatsapp.send_message(to=m["customer_phone"], text=msg)
                await database.mark_link_sent(m["id"])
                logger.info(
                    "[INFO] [main] Meeting link sent — phone: %s, meeting_id: %s",
                    mask_phone(m["customer_phone"]),
                    m["id"],
                )
        except Exception as exc:
            logger.error("[ERROR] [main] Link delivery job error: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Creates DB pool on startup, starts link delivery job, closes on shutdown."""
    logger.info("[INFO] [main] Starting up — creating database connection pool")
    await database.create_pool()
    logger.info("[INFO] [main] Database pool ready")
    delivery_task = asyncio.create_task(_link_delivery_loop())
    logger.info("[INFO] [main] Meeting link delivery job started")
    yield
    delivery_task.cancel()
    try:
        await delivery_task
    except asyncio.CancelledError:
        pass
    logger.info("[INFO] [main] Shutting down — closing database connection pool")
    await database.close_pool()
    logger.info("[INFO] [main] Database pool closed")


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Webhook verification
# ---------------------------------------------------------------------------


@app.get("/webhook")
async def verify_webhook(request: Request):
    """Meta webhook verification endpoint."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    logger.info("[INFO] [main] Webhook verification request — mode: %s", mode)

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("[INFO] [main] Webhook verified successfully")
        return PlainTextResponse(content=challenge, status_code=200)

    logger.warning("[WARN] [main] Webhook verification failed — token mismatch")
    return PlainTextResponse(content="Forbidden", status_code=403)


# ---------------------------------------------------------------------------
# Incoming message webhook
# ---------------------------------------------------------------------------


@app.post("/webhook")
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    """
    Meta incoming-message webhook.
    Returns 200 immediately, processes in background to prevent Meta retries.

    Resolves company_id from the WhatsApp phone_number_id in the webhook metadata.
    Falls back to company_id=1 if the phone number is not yet registered.
    """
    body = await request.json()

    try:
        entry = body.get("entry", [])
        changes = entry[0].get("changes", []) if entry else []
        value = changes[0].get("value", {}) if changes else {}

        messages_list = value.get("messages", [])
        if not messages_list:
            logger.info("[INFO] [main] No messages in payload — likely a status update, ignoring")
            return JSONResponse(content={"status": "ok"}, status_code=200)

        # Resolve company from the WhatsApp phone_number_id in the webhook metadata.
        phone_number_id = value.get("metadata", {}).get("phone_number_id", "")
        company_id = await database.get_company_by_phone_number_id(phone_number_id)

        message = messages_list[0]
        msg_type = message.get("type")
        customer_phone = message.get("from")

        if not customer_phone:
            logger.warning("[WARN] [main] Webhook message missing 'from' field — ignoring")
            return JSONResponse(content={"status": "ok"}, status_code=200)

        if msg_type == "text":
            message_text = message.get("text", {}).get("body")
            if not message_text:
                logger.warning(
                    "[WARN] [main] Text message with empty body — phone: %s",
                    mask_phone(customer_phone),
                )
                return JSONResponse(content={"status": "ok"}, status_code=200)
            logger.info(
                "[INFO] [main] Message received — phone: %s, type: text, company_id: %d",
                mask_phone(customer_phone),
                company_id,
            )
            background_tasks.add_task(process_message, customer_phone, message_text, company_id)

        elif msg_type == "audio":
            audio_data = message.get("audio", {})
            media_id = audio_data.get("id")
            mime_type = audio_data.get("mime_type", "audio/ogg")
            if not media_id:
                logger.warning(
                    "[WARN] [main] Audio message missing media ID — phone: %s",
                    mask_phone(customer_phone),
                )
                return JSONResponse(content={"status": "ok"}, status_code=200)
            logger.info(
                "[INFO] [main] Message received — phone: %s, type: audio, mime: %s, company_id: %d",
                mask_phone(customer_phone),
                mime_type,
                company_id,
            )
            background_tasks.add_task(
                process_audio_message, customer_phone, media_id, mime_type, company_id
            )

        else:
            logger.info(
                "[INFO] [main] Unsupported message type — phone: %s, type: %s",
                mask_phone(customer_phone),
                msg_type,
            )

    except (IndexError, KeyError, TypeError) as exc:
        logger.error("[ERROR] [main] Failed to parse webhook payload: %s", exc, exc_info=True)

    return JSONResponse(content={"status": "ok"}, status_code=200)


# ---------------------------------------------------------------------------
# Agent send endpoint (called by the dashboard)
# ---------------------------------------------------------------------------


@app.post("/send")
async def send_agent_message(request: Request):
    """
    Called by the agent dashboard when an agent sends a manual reply.
    Validates the webhook secret, sends via WhatsApp, saves with sender='agent'.
    """
    secret = request.headers.get("x-webhook-secret")
    if secret != WEBHOOK_SECRET:
        logger.warning("[WARN] [main] /send rejected — invalid webhook secret")
        return JSONResponse(content={"error": "Forbidden"}, status_code=403)

    body = await request.json()
    customer_phone = body.get("customer_phone")
    message_text = body.get("message")
    company_id = int(body.get("company_id", 1))

    if not customer_phone or not message_text:
        return JSONResponse(content={"error": "Missing fields"}, status_code=400)

    try:
        await whatsapp.send_message(to=customer_phone, text=message_text)
        await memory.save_message(
            customer_phone=customer_phone,
            direction="outbound",
            message_text=message_text,
            sender="agent",
            company_id=company_id,
        )
        logger.info(
            "[INFO] [main] Agent message sent — phone: %s, type: text",
            mask_phone(customer_phone),
        )
        return JSONResponse(content={"status": "sent"}, status_code=200)
    except Exception as exc:
        logger.error(
            "[ERROR] [main] Failed to send agent message — phone: %s, error: %s",
            mask_phone(customer_phone),
            exc,
            exc_info=True,
        )
        return JSONResponse(content={"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------


async def process_message(customer_phone: str, message_text: str, company_id: int = 1):
    """Generate and send a bot reply for an inbound text message."""
    try:
        logger.info(
            "[INFO] [main] Processing text message — phone: %s",
            mask_phone(customer_phone),
        )

        reply, meeting_message = await agent.get_reply(
            customer_phone=customer_phone,
            new_message=message_text,
            company_id=company_id,
        )

        await whatsapp.send_message(to=customer_phone, text=reply)
        logger.info(
            "[INFO] [main] Reply sent — phone: %s, type: text",
            mask_phone(customer_phone),
        )

        if meeting_message:
            await whatsapp.send_message(to=customer_phone, text=meeting_message)
            await memory.save_message(
                customer_phone=customer_phone,
                direction="outbound",
                message_text=meeting_message,
                sender="ai",
                company_id=company_id,
            )
            logger.info(
                "[INFO] [main] Meeting invitation sent — phone: %s",
                mask_phone(customer_phone),
            )

    except Exception as exc:
        logger.error(
            "[ERROR] [main] Failed to process text message — phone: %s, error: %s",
            mask_phone(customer_phone),
            exc,
            exc_info=True,
        )


async def process_audio_message(customer_phone: str, media_id: str, mime_type: str, company_id: int = 1):
    """
    Handle an incoming WhatsApp voice note end-to-end:
    1. Download audio from Meta's CDN.
    2. Store audio bytes in the voice_notes table.
    3. Transcribe with OpenAI Whisper.
    4. Save the inbound message with media metadata.
    5. Feed the transcription into the normal bot flow.
    """
    try:
        logger.info(
            "[INFO] [main] Processing voice note — phone: %s, mime: %s",
            mask_phone(customer_phone),
            mime_type,
        )

        # Step 1: Download
        try:
            audio_bytes, actual_mime = await transcribe_mod.download_media(media_id)
        except ValueError as exc:
            if "too large" in str(exc).lower():
                logger.warning(
                    "[WARN] [main] Voice note rejected (too large) — phone: %s",
                    mask_phone(customer_phone),
                )
                await whatsapp.send_message(
                    to=customer_phone,
                    text=(
                        "Sorry, your voice message is too long for me to process. "
                        "Could you send a shorter message (under 3 minutes) "
                        "or type your question instead?"
                    ),
                )
            else:
                raise
            return

        # Step 2: Store audio in DB
        audio_id = await database.store_voice_note(audio_bytes, actual_mime)
        raw_base = (os.environ.get("APP_URL") or "wak-agents.up.railway.app").rstrip("/")
        base_url = raw_base if raw_base.startswith("http") else f"https://{raw_base}"
        media_url = f"{base_url}/api/voice-notes/{audio_id}"

        # Step 3: Transcribe
        try:
            transcription = await transcribe_mod.transcribe(audio_bytes, actual_mime)
        except Exception as exc:
            logger.error(
                "[ERROR] [main] Whisper transcription failed — phone: %s, error: %s",
                mask_phone(customer_phone),
                exc,
                exc_info=True,
            )
            await memory.save_message(
                customer_phone=customer_phone,
                direction="inbound",
                message_text="[Voice message — transcription unavailable]",
                sender="customer",
                media_type="audio",
                media_url=media_url,
                transcription=None,
                company_id=company_id,
            )
            await whatsapp.send_message(
                to=customer_phone,
                text=(
                    "Sorry, I couldn't process your voice message. "
                    "Could you type your question instead?"
                ),
            )
            return

        if not transcription:
            logger.info(
                "[INFO] [main] Whisper returned empty transcription — phone: %s",
                mask_phone(customer_phone),
            )
            await memory.save_message(
                customer_phone=customer_phone,
                direction="inbound",
                message_text="[Voice message — no speech detected]",
                sender="customer",
                media_type="audio",
                media_url=media_url,
                transcription="",
                company_id=company_id,
            )
            await whatsapp.send_message(
                to=customer_phone,
                text=(
                    "I received your voice message but couldn't make out any words. "
                    "Could you type your question instead?"
                ),
            )
            return

        # Step 4: Save inbound with full media metadata
        await memory.save_message(
            customer_phone=customer_phone,
            direction="inbound",
            message_text=transcription,
            sender="customer",
            media_type="audio",
            media_url=media_url,
            transcription=transcription,
            company_id=company_id,
        )

        # Step 5: Run transcription through the normal bot flow
        reply, meeting_message = await agent.get_reply(
            customer_phone=customer_phone,
            new_message=transcription,
            _save_inbound=False,
            company_id=company_id,
        )

        await whatsapp.send_message(to=customer_phone, text=reply)
        logger.info(
            "[INFO] [main] Reply sent after voice note — phone: %s, type: text",
            mask_phone(customer_phone),
        )

        if meeting_message:
            await whatsapp.send_message(to=customer_phone, text=meeting_message)
            await memory.save_message(
                customer_phone=customer_phone,
                direction="outbound",
                message_text=meeting_message,
                sender="ai",
                company_id=company_id,
            )

    except Exception as exc:
        logger.error(
            "[ERROR] [main] Failed to process voice note — phone: %s, error: %s",
            mask_phone(customer_phone),
            exc,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Audio streaming endpoint
# ---------------------------------------------------------------------------


@app.get("/audio/{audio_id}")
async def serve_audio(audio_id: str):
    """
    Stream a stored voice note. The UUID acts as a capability token —
    no additional auth needed.
    """
    row = await database.get_voice_note(audio_id)
    if row is None:
        logger.warning("[WARN] [main] Audio not found — id: %s", audio_id)
        return JSONResponse(content={"error": "Not found"}, status_code=404)
    return Response(
        content=row["audio_data"],
        media_type=row["mime_type"],
        headers={
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
        },
    )
