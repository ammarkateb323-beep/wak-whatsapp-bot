"""
database.py — asyncpg connection pool and all SQL query functions.

Multi-tenancy: every write/read is scoped to a company_id.
Use get_company_by_phone_number_id() at the start of every inbound
message handler to resolve the company from the WhatsApp phone number ID.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone

import asyncpg

from config import DATABASE_URL

logger = logging.getLogger(__name__)

# Holds the pool once created by main.py on startup.
pool: asyncpg.Pool | None = None

# In-process cache: phone_number_id → company_id (avoids a DB round-trip per message)
_company_cache: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Pool lifecycle
# ---------------------------------------------------------------------------


async def create_pool() -> None:
    """
    Opens a pool of 2–10 reusable connections to Neon.
    Called once on FastAPI startup in main.py.
    Strips query parameters from the URL because asyncpg doesn't support them.
    """
    global pool
    clean_url = DATABASE_URL.split("?")[0]
    try:
        pool = await asyncpg.create_pool(
            dsn=clean_url,
            ssl="require",
            min_size=2,
            max_size=10,
        )
        logger.info("[INFO] [database] Connection pool created — min: 2, max: 10")
    except Exception as exc:
        logger.error("[ERROR] [database] Failed to create connection pool: %s", exc, exc_info=True)
        raise


async def close_pool() -> None:
    """Cleanly closes all pool connections on FastAPI shutdown."""
    global pool
    if pool:
        await pool.close()
        logger.info("[INFO] [database] Connection pool closed")


# ---------------------------------------------------------------------------
# Multi-tenancy: company resolution
# ---------------------------------------------------------------------------


async def get_company_by_phone_number_id(phone_number_id: str) -> int:
    """
    Looks up the company_id that owns this WhatsApp phone_number_id.
    Results are cached in-process so only the first message per process
    triggers a DB round-trip.

    Falls back to company_id = 1 if no match is found (safe during migration
    while existing companies haven't had their whatsapp_phone_number_id set).
    """
    if phone_number_id in _company_cache:
        return _company_cache[phone_number_id]

    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id FROM companies
                WHERE whatsapp_phone_number_id = $1 AND is_active = true
                LIMIT 1
                """,
                phone_number_id,
            )
        company_id = row["id"] if row else 1
        if not row:
            logger.warning(
                "[WARN] [database] No company found for phone_number_id %s — falling back to company_id=1",
                phone_number_id,
            )
        else:
            logger.info(
                "[INFO] [database] Resolved company_id=%d for phone_number_id=%s",
                company_id,
                phone_number_id,
            )
        _company_cache[phone_number_id] = company_id
        return company_id
    except Exception as exc:
        logger.error(
            "[ERROR] [database] get_company_by_phone_number_id failed: %s", exc, exc_info=True
        )
        return 1  # safe fallback — never crash an inbound message


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


async def lookup_order(order_number: str, company_id: int = 1) -> dict:
    """
    Query the orders table for a given order_number scoped to company_id.
    Called by agent.py when OpenAI triggers the lookup_order tool.
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT order_number, status, details, created_at
                FROM orders
                WHERE order_number = $1 AND company_id = $2
                """,
                order_number,
                company_id,
            )
        if row is None:
            logger.info("[INFO] [database] Order not found — order_number: %s", order_number)
            return {"found": False, "message": f"No order found with number {order_number}."}
        logger.info(
            "[INFO] [database] Order lookup success — order_number: %s, status: %s",
            order_number,
            row["status"],
        )
        return {
            "found": True,
            "order_number": row["order_number"],
            "status": row["status"],
            "details": row["details"],
            "created_at": str(row["created_at"]),
        }
    except Exception as exc:
        logger.error(
            "[ERROR] [database] lookup_order failed — order_number: %s, error: %s",
            order_number,
            exc,
            exc_info=True,
        )
        raise


# ---------------------------------------------------------------------------
# Meetings
# ---------------------------------------------------------------------------


async def create_meeting_with_token(customer_phone: str, company_id: int = 1) -> str:
    """
    Creates a meeting record with a unique booking token (24 hr expiry).
    Returns the token UUID string.
    """
    token = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO meetings
                  (customer_phone, meeting_link, meeting_token, token_expires_at, status, company_id, created_at)
                VALUES ($1, '', $2, $3, 'pending', $4, NOW())
                """,
                customer_phone,
                token,
                expires_at,
                company_id,
            )
        logger.info("[INFO] [database] Meeting token created — token: %s", token[:8] + "...")
        return token
    except Exception as exc:
        logger.error("[ERROR] [database] create_meeting_with_token failed: %s", exc, exc_info=True)
        raise


async def get_pending_meeting(customer_phone: str, company_id: int = 1) -> dict | None:
    """
    Returns the latest pending meeting for a customer within the given company, or None.
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, meeting_link, agreed_time, meeting_token, scheduled_at
                FROM meetings
                WHERE customer_phone = $1 AND status = 'pending' AND company_id = $2
                ORDER BY created_at DESC
                LIMIT 1
                """,
                customer_phone,
                company_id,
            )
        if row is None:
            return None
        return {
            "id": row["id"],
            "meeting_link": row["meeting_link"],
            "agreed_time": row["agreed_time"],
            "meeting_token": row["meeting_token"],
            "scheduled_at": row["scheduled_at"],
        }
    except Exception as exc:
        logger.error(
            "[ERROR] [database] get_pending_meeting failed: %s", exc, exc_info=True
        )
        raise


async def update_meeting_time(meeting_id: int, agreed_time: str) -> None:
    """Saves the agreed date/time for a pending meeting."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE meetings SET agreed_time = $1 WHERE id = $2",
                agreed_time,
                meeting_id,
            )
        logger.info(
            "[INFO] [database] Meeting time updated — meeting_id: %d, agreed_time: %s",
            meeting_id,
            agreed_time,
        )
    except Exception as exc:
        logger.error(
            "[ERROR] [database] update_meeting_time failed — meeting_id: %d, error: %s",
            meeting_id,
            exc,
            exc_info=True,
        )
        raise


async def get_meetings_to_notify() -> list[dict]:
    """
    Returns meetings within 15 minutes of start time that haven't had
    their link sent yet.
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, customer_phone, meeting_link, meeting_token
                FROM meetings
                WHERE status != 'completed'
                  AND link_sent = FALSE
                  AND meeting_link != ''
                  AND scheduled_at IS NOT NULL
                  AND scheduled_at <= NOW() + INTERVAL '15 minutes'
                  AND scheduled_at >= NOW() - INTERVAL '30 minutes'
                """
            )
        result = [dict(r) for r in rows]
        if result:
            logger.info(
                "[INFO] [database] Meetings to notify — count: %d", len(result)
            )
        return result
    except Exception as exc:
        logger.error(
            "[ERROR] [database] get_meetings_to_notify failed: %s", exc, exc_info=True
        )
        raise


async def mark_link_sent(meeting_id: int) -> None:
    """Marks the meeting link as sent to prevent duplicate delivery."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE meetings SET link_sent = TRUE WHERE id = $1",
                meeting_id,
            )
    except Exception as exc:
        logger.error(
            "[ERROR] [database] mark_link_sent failed — meeting_id: %d, error: %s",
            meeting_id,
            exc,
            exc_info=True,
        )
        raise


# ---------------------------------------------------------------------------
# Voice notes
# ---------------------------------------------------------------------------


async def store_voice_note(audio_bytes: bytes, mime_type: str) -> str:
    """
    Persist voice note audio in the voice_notes table.
    Returns the UUID string that identifies this recording.
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO voice_notes (audio_data, mime_type)
                VALUES ($1, $2)
                RETURNING id::text
                """,
                audio_bytes,
                mime_type,
            )
        audio_id = row["id"]
        logger.info(
            "[INFO] [database] Voice note stored — id: %s, size_bytes: %d, mime: %s",
            audio_id[:8] + "...",
            len(audio_bytes),
            mime_type,
        )
        return audio_id
    except Exception as exc:
        logger.error(
            "[ERROR] [database] store_voice_note failed — mime: %s, error: %s",
            mime_type,
            exc,
            exc_info=True,
        )
        raise


async def get_voice_note(audio_id: str) -> dict | None:
    """
    Fetch a voice note by its UUID.
    Returns dict with 'audio_data' and 'mime_type', or None if not found.
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT audio_data, mime_type FROM voice_notes WHERE id = $1::uuid",
                audio_id,
            )
        if row is None:
            logger.warning("[WARN] [database] Voice note not found — id: %s", audio_id)
            return None
        return {"audio_data": bytes(row["audio_data"]), "mime_type": row["mime_type"]}
    except Exception as exc:
        logger.error(
            "[ERROR] [database] get_voice_note failed — id: %s, error: %s",
            audio_id,
            exc,
            exc_info=True,
        )
        raise


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------


async def auto_capture_contact(customer_phone: str, company_id: int = 1) -> None:
    """
    Upsert a contact row for a customer who just sent an inbound message.
    Scoped to company_id. Does nothing if the number already exists for this company.
    """
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO contacts (phone_number, source, company_id)
                VALUES ($1, 'whatsapp', $2)
                ON CONFLICT (phone_number) DO NOTHING
                """,
                customer_phone,
                company_id,
            )
    except Exception as exc:
        logger.error(
            "[ERROR] [database] auto_capture_contact failed: %s", exc, exc_info=True
        )
        # Non-fatal — don't raise, just log.


# ---------------------------------------------------------------------------
# Escalations
# ---------------------------------------------------------------------------


async def create_escalation(customer_phone: str, escalation_reason: str, company_id: int = 1) -> None:
    """
    Inserts or updates an escalation record for a customer, scoped to company_id.
    Uses ON CONFLICT to avoid duplicates if the customer escalates twice.
    """
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO escalations (customer_phone, escalation_reason, status, company_id, created_at)
                VALUES ($1, $2, 'open', $3, NOW())
                ON CONFLICT (customer_phone)
                DO UPDATE SET
                    escalation_reason = EXCLUDED.escalation_reason,
                    status = 'open',
                    created_at = NOW()
                """,
                customer_phone,
                escalation_reason,
                company_id,
            )
        logger.info("[INFO] [database] Escalation created/updated — company_id: %d", company_id)
    except Exception as exc:
        logger.error(
            "[ERROR] [database] create_escalation failed: %s", exc, exc_info=True
        )
        raise
