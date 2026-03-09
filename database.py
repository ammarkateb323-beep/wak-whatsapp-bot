import asyncpg
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
