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

# Basic logging setup so we can see what's happening in the terminal.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def _link_delivery_loop():
    """
    Background task that runs every 60 seconds.
    Sends Jitsi meeting links to customers whose meetings start within 15 minutes.
    """
    while True:
        await asyncio.sleep(60)
        try:
            meetings = await database.get_meetings_to_notify()
            for m in meetings:
                raw_base = (os.environ.get("APP_URL") or "wak-agents.up.railway.app").rstrip("/")
                base_url = raw_base if raw_base.startswith("http") else f"https://{raw_base}"
                meeting_url = f"{base_url}/meeting/{m['meeting_token']}" if m.get("meeting_token") else m["meeting_link"]
                msg = f"Your meeting is starting soon! Join here: {meeting_url}"
                await whatsapp.send_message(to=m["customer_phone"], text=msg)
                await database.mark_link_sent(m["id"])
                logger.info("Sent meeting link to %s (meeting %s)", m["customer_phone"], m["id"])
        except Exception as exc:
            logger.error("Link delivery job error: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    App startup/shutdown lifecycle.
    Creates DB pool on startup, starts link delivery job, closes on shutdown.
    """
    logger.info("Starting up - creating database connection pool...")
    await database.create_pool()
    logger.info("Database pool ready.")
    delivery_task = asyncio.create_task(_link_delivery_loop())
    logger.info("Meeting link delivery job started.")
    yield
    delivery_task.cancel()
    try:
        await delivery_task
    except asyncio.CancelledError:
        pass
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
        msg_type = message.get("type")
        customer_phone = message.get("from")

        if not customer_phone:
            logger.warning("Webhook message missing 'from' field. Ignoring.")
            return JSONResponse(content={"status": "ok"}, status_code=200)

        if msg_type == "text":
            message_text = message.get("text", {}).get("body")
            if not message_text:
                logger.warning("Text message with empty body from %s. Ignoring.", customer_phone)
                return JSONResponse(content={"status": "ok"}, status_code=200)
            logger.info("Text message from %s: %s", customer_phone, message_text[:50])
            background_tasks.add_task(process_message, customer_phone, message_text)

        elif msg_type == "audio":
            audio_data = message.get("audio", {})
            media_id   = audio_data.get("id")
            mime_type  = audio_data.get("mime_type", "audio/ogg")
            if not media_id:
                logger.warning("Audio message from %s missing media ID. Ignoring.", customer_phone)
                return JSONResponse(content={"status": "ok"}, status_code=200)
            logger.info("Voice note from %s — media_id=%s", customer_phone, media_id)
            background_tasks.add_task(process_audio_message, customer_phone, media_id, mime_type)

        else:
            logger.info(
                "Unsupported message type '%s' from %s. Ignoring.",
                msg_type, customer_phone,
            )

    except (IndexError, KeyError, TypeError) as exc:
        logger.error("Failed to parse webhook payload: %s", exc)
        # Always return 200 to prevent retry loops from Meta.
        return JSONResponse(content={"status": "ok"}, status_code=200)

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

        reply, meeting_message = await agent.get_reply(
            customer_phone=customer_phone,
            new_message=message_text,
        )
        logger.info("Reply generated for %s: %s...", customer_phone, reply[:50])

        await whatsapp.send_message(to=customer_phone, text=reply)
        logger.info("Reply sent to %s successfully.", customer_phone)

        # If a meeting was just created, send the invitation as a separate message.
        if meeting_message:
            await whatsapp.send_message(to=customer_phone, text=meeting_message)
            await memory.save_message(
                customer_phone=customer_phone,
                direction="outbound",
                message_text=meeting_message,
                sender="ai",
            )
            logger.info("Meeting invitation sent to %s.", customer_phone)

    except Exception as exc:
        logger.error(
            "Failed to process message from %s: %s",
            customer_phone,
            exc,
            exc_info=True,
        )


async def process_audio_message(customer_phone: str, media_id: str, mime_type: str):
    """
    Handle an incoming WhatsApp voice note end-to-end:

    1. Download audio from Meta's CDN.
    2. Store audio bytes in the voice_notes table.
    3. Transcribe with OpenAI Whisper.
    4. Save the inbound message with media metadata.
    5. Feed the transcription into the normal bot flow.

    If transcription fails or the file is too large, we reply asking
    the customer to type their message instead.
    """
    try:
        logger.info("Processing voice note from %s (media_id=%s)...", customer_phone, media_id)

        # ── Step 1: Download ──────────────────────────────────────────────
        try:
            audio_bytes, actual_mime = await transcribe_mod.download_media(media_id)
        except ValueError as exc:
            if "too large" in str(exc).lower():
                logger.warning("Voice note from %s rejected: %s", customer_phone, exc)
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

        # ── Step 2: Store audio in DB ─────────────────────────────────────
        audio_id = await database.store_voice_note(audio_bytes, actual_mime)

        # Build the playback URL using APP_URL (same env var used elsewhere)
        raw_base = (os.environ.get("APP_URL") or "").rstrip("/")
        if raw_base and not raw_base.startswith("http"):
            raw_base = f"https://{raw_base}"
        media_url = f"{raw_base}/audio/{audio_id}" if raw_base else None

        # ── Step 3: Transcribe ────────────────────────────────────────────
        try:
            transcription = await transcribe_mod.transcribe(audio_bytes, actual_mime)
        except Exception as exc:
            logger.error("Whisper failed for %s: %s", customer_phone, exc, exc_info=True)
            # Save the voice note without transcription so agents can still play it
            await memory.save_message(
                customer_phone=customer_phone,
                direction="inbound",
                message_text="[Voice message — transcription unavailable]",
                sender="customer",
                media_type="audio",
                media_url=media_url,
                transcription=None,
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
            logger.info("Whisper returned empty transcription for %s.", customer_phone)
            await memory.save_message(
                customer_phone=customer_phone,
                direction="inbound",
                message_text="[Voice message — no speech detected]",
                sender="customer",
                media_type="audio",
                media_url=media_url,
                transcription="",
            )
            await whatsapp.send_message(
                to=customer_phone,
                text=(
                    "I received your voice message but couldn't make out any words. "
                    "Could you type your question instead?"
                ),
            )
            return

        logger.info(
            "Voice note transcribed for %s: %r...",
            customer_phone, transcription[:60],
        )

        # ── Step 4: Save inbound with full media metadata ─────────────────
        await memory.save_message(
            customer_phone=customer_phone,
            direction="inbound",
            message_text=transcription,   # bot reads this as the customer's text
            sender="customer",
            media_type="audio",
            media_url=media_url,
            transcription=transcription,
        )

        # ── Step 5: Run the transcription through the normal bot flow ─────
        # _save_inbound=False because we already saved the inbound message above.
        reply, meeting_message = await agent.get_reply(
            customer_phone=customer_phone,
            new_message=transcription,
            _save_inbound=False,
        )

        await whatsapp.send_message(to=customer_phone, text=reply)
        logger.info("Reply sent to %s after voice note.", customer_phone)

        if meeting_message:
            await whatsapp.send_message(to=customer_phone, text=meeting_message)
            await memory.save_message(
                customer_phone=customer_phone,
                direction="outbound",
                message_text=meeting_message,
                sender="ai",
            )

    except Exception as exc:
        logger.error(
            "Failed to process voice note from %s: %s",
            customer_phone,
            exc,
            exc_info=True,
        )


@app.get("/audio/{audio_id}")
async def serve_audio(audio_id: str):
    """
    Stream a stored voice note back to the caller.

    The agent dashboard loads audio from this endpoint via the media_url
    stored in the messages table.  UUIDs are hard to guess, so there is
    no additional authentication — the URL itself acts as a capability token.
    """
    row = await database.get_voice_note(audio_id)
    if row is None:
        return JSONResponse(content={"error": "Not found"}, status_code=404)
    return Response(
        content=row["audio_data"],
        media_type=row["mime_type"],
        headers={
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",   # allow browser to load cross-origin
        },
    )
