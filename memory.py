import database
from config import MEMORY_WINDOW


async def load_history(customer_phone: str) -> list[dict]:
    """
    Loads the last N messages for a customer from the messages table
    and returns them formatted as OpenAI conversation history.

    Called by agent.py before every OpenAI API call.

    Args:
        customer_phone: The customer's WhatsApp number (e.g. "971501234567")

    Returns:
        A list of dicts in OpenAI's format:
        [
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Welcome to WAK..."},
            ...
        ]
        Returns an empty list [] if this is a brand new customer —
        which tells the agent this is a fresh conversation and it must
        send the STEP 0 opening message.
    """
    async with database.pool.acquire() as conn:
        # We fetch the last MEMORY_WINDOW messages ordered by created_at.
        #
        # The inner query (subquery) gets the most recent N rows in
        # reverse order (DESC = newest first).
        #
        # The outer query re-sorts them oldest-first (ASC) so OpenAI
        # receives the conversation in the correct chronological order.
        #
        # Why a subquery? If we just did ORDER BY created_at ASC LIMIT 20,
        # we'd get the oldest 20 messages, not the most recent 20.
        rows = await conn.fetch(
            """
            SELECT role, message_text
            FROM (
                SELECT sender     AS role,
                       message_text,
                       created_at
                FROM   messages
                WHERE  customer_phone = $1
                ORDER  BY created_at DESC
                LIMIT  $2
            ) recent_messages
            ORDER BY created_at ASC
            """,
            customer_phone,  # $1 — filters to this customer only
            MEMORY_WINDOW,   # $2 — how many messages to load (20 from config.py)
        )

    # If no rows returned, this is a new customer — return empty list.
    # agent.py will see an empty history and know to send the opening message.
    if not rows:
        return []

    # Convert asyncpg Row objects into plain dicts that OpenAI accepts.
    # Each row becomes: {"role": "user"|"assistant", "content": "..."}
    return [
        {
            "role": (
                "user"
                if row["role"] == "customer"
                else "assistant"
            ),
            "content": row["message_text"],
        }
        for row in rows
    ]


async def save_message(
    customer_phone: str,
    direction: str,
    message_text: str,
    sender: str = None,
    media_type: str = None,
    media_url: str = None,
    transcription: str = None,
):
    """
    Saves a single message to the messages table.

    Called after every exchange to save the customer's message and the
    bot's reply. The optional sender override is used for human agent
    replies coming from the dashboard.

    For voice notes, pass media_type="audio", media_url=<playback URL>,
    and transcription=<Whisper text>. message_text should also be set to
    the transcription so load_history() continues to work unchanged.

    Args:
        customer_phone: The customer's WhatsApp number
        direction:      "inbound" or "outbound"
        message_text:   Text content (transcription for voice notes)
        sender:         Optional explicit sender override
        media_type:     "audio" for voice notes, None for text
        media_url:      URL to the stored audio file (None for text)
        transcription:  Whisper output text (None for text messages)
    """
    if sender is None:
        sender = "customer" if direction == "inbound" else "ai"

    async with database.pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO messages
              (customer_phone, direction, sender, message_text,
               media_type, media_url, transcription, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            """,
            customer_phone,
            direction,
            sender,
            message_text,
            media_type,
            media_url,
            transcription,
        )
        # NOW() is a PostgreSQL function that inserts the current timestamp.
        # We let the DB set the time rather than Python to avoid timezone
        # mismatches between your server and the Neon database.
