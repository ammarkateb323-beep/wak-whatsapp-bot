import asyncpg
import random
import string
import uuid
from datetime import datetime, timezone, timedelta
from config import DATABASE_URL


# This variable holds the pool once it's created.
# It starts as None — main.py will initialise it on startup.
pool: asyncpg.Pool | None = None


async def create_pool():
    """
    Called once when FastAPI starts up (in main.py).
    Opens a pool of 2-10 reusable connections to Neon.

    We strip the query parameters from the URL (?sslmode=require...)
    because asyncpg doesn't support them as URL params — instead we
    pass ssl="require" as a keyword argument directly.
    """
    global pool

    # Strip query parameters from the Neon URL.
    # Neon gives you: postgresql://user:pass@host/db?sslmode=require&channel_binding=require
    # asyncpg needs:  postgresql://user:pass@host/db  (+ ssl handled separately)
    clean_url = DATABASE_URL.split("?")[0]

    pool = await asyncpg.create_pool(
        dsn=clean_url,  # The cleaned connection string
        ssl="require",  # Neon requires SSL — passed directly to asyncpg
        min_size=2,     # Keep at least 2 connections open at all times
        max_size=10,    # Allow up to 10 simultaneous connections under load
    )


async def close_pool():
    """
    Called once when FastAPI shuts down (in main.py).
    Cleanly closes all connections in the pool.
    Not strictly required but good practice — avoids leaving
    dangling connections on the Neon side.
    """
    global pool
    if pool:
        await pool.close()


async def lookup_order(order_number: str) -> dict:
    """
    Queries the orders table for a given order_number.
    Called by agent.py when OpenAI triggers the lookup_order tool.

    Returns a dict with the order details, or a not-found message.
    The result gets sent back to OpenAI so it can write a natural reply.

    Args:
        order_number: The order number the customer provided.

    Returns:
        dict: Either the order data, or a message saying it wasn't found.
    """
    # Acquire a connection from the pool.
    # The "async with" block automatically returns it to the pool when done.
    async with pool.acquire() as conn:
        # $1 is asyncpg's placeholder for the first argument (order_number).
        # This is parameterised — never use f-strings for SQL, that's how
        # SQL injection attacks happen. asyncpg handles escaping safely.
        row = await conn.fetchrow(
            """
            SELECT order_number, status, details, created_at
            FROM orders
            WHERE order_number = $1
            """,
            order_number,  # asyncpg maps this to $1
        )

    # fetchrow() returns None if no matching row was found.
    if row is None:
        return {
            "found": False,
            "message": f"No order found with number {order_number}.",
        }

    # Convert the asyncpg Record object to a plain dict.
    # We also convert created_at to a string so it's JSON-serialisable
    # (datetime objects can't be sent to OpenAI as-is).
    return {
        "found": True,
        "order_number": row["order_number"],
        "status": row["status"],
        "details": row["details"],
        "created_at": str(row["created_at"]),
    }


async def create_meeting_with_token(customer_phone: str) -> str:
    """
    Creates a meeting record with a unique booking token.
    The Jitsi link is generated later when the customer books a slot.
    Returns the booking token (UUID).
    """
    token = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO meetings
              (customer_phone, meeting_link, meeting_token, token_expires_at, status, created_at)
            VALUES ($1, '', $2, $3, 'pending', NOW())
            """,
            customer_phone,
            token,
            expires_at,
        )
    return token


async def get_pending_meeting(customer_phone: str) -> dict | None:
    """
    Returns the latest pending meeting for a customer, or None.
    Includes meeting_token and scheduled_at so agent.py can resend
    the booking URL if the meeting is still unbooked (scheduled_at IS NULL).
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, meeting_link, agreed_time, meeting_token, scheduled_at
            FROM meetings
            WHERE customer_phone = $1 AND status = 'pending'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            customer_phone,
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


async def update_meeting_time(meeting_id: int, agreed_time: str) -> None:
    """
    Saves the date/time the customer agreed to for their pending meeting.
    """
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE meetings SET agreed_time = $1 WHERE id = $2",
            agreed_time,
            meeting_id,
        )


async def get_meetings_to_notify() -> list[dict]:
    """
    Returns meetings that are within 15 minutes of their scheduled time,
    have not had their link sent yet, and are not completed.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, customer_phone, meeting_link
            FROM meetings
            WHERE status != 'completed'
              AND link_sent = FALSE
              AND meeting_link != ''
              AND scheduled_at IS NOT NULL
              AND scheduled_at <= NOW() + INTERVAL '15 minutes'
              AND scheduled_at >= NOW() - INTERVAL '30 minutes'
            """
        )
    return [dict(r) for r in rows]


async def mark_link_sent(meeting_id: int) -> None:
    """Marks the Jitsi link as sent so it is not sent again."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE meetings SET link_sent = TRUE WHERE id = $1",
            meeting_id,
        )


async def create_escalation(customer_phone: str, escalation_reason: str):
    """
    Inserts or updates an escalation record for a customer.
    Called when the AI sends a handoff phrase.
    Uses ON CONFLICT to avoid duplicates if the customer escalates twice.
    """
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO escalations (customer_phone, escalation_reason, status, created_at)
            VALUES ($1, $2, 'open', NOW())
            ON CONFLICT (customer_phone)
            DO UPDATE SET
                escalation_reason = EXCLUDED.escalation_reason,
                status = 'open',
                created_at = NOW()
            """,
            customer_phone,
            escalation_reason,
        )
