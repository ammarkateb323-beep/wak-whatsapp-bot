import os
from dotenv import load_dotenv

# load_dotenv() finds the .env file in your project root and loads
# every key=value pair into the environment. This must happen before
# any os.getenv() calls below — so it sits at the very top.
load_dotenv()


# ── OpenAI ────────────────────────────────────────────────────────────

# The secret key that authenticates your requests to OpenAI.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# The model that handles conversations. gpt-4.1-mini is fast, cheap,
# and fully supports tool calling — which we need for order lookup.
# To upgrade to gpt-5-mini later, change only this one line.
OPENAI_MODEL = "gpt-4.1-mini"


# ── Neon (PostgreSQL) ─────────────────────────────────────────────────

# Full connection string to your Neon database.
# asyncpg will use this to open a connection pool when the app starts.
# Format: postgresql://user:password@host/dbname
DATABASE_URL = os.getenv("DATABASE_URL")


# ── Meta WhatsApp Cloud API ───────────────────────────────────────────

# Permanent access token from your Meta developer dashboard.
# This authorises your server to send WhatsApp messages.
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")

# The phone number ID — not the phone number itself.
# Meta uses this to identify which WhatsApp number is sending the message.
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")


# ── Webhook Verification ──────────────────────────────────────────────

# When you register your server URL with Meta, they send a GET request
# containing this token to confirm you own and control the server.
# You choose this string — it just needs to match on both sides.
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")

DASHBOARD_URL = os.getenv("DASHBOARD_URL")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")


# ── Memory Settings ───────────────────────────────────────────────────

# How many past messages to load from the messages table and pass to
# OpenAI as conversation history. 20 messages = roughly 10 back-and-forth
# exchanges. Enough context without burning unnecessary tokens.
MEMORY_WINDOW = 20


# ── Startup Validation ────────────────────────────────────────────────

# If any required variable is missing from .env, we crash immediately
# on startup with a clear error — not silently later when a customer
# sends a message and something mysteriously fails.
_required = {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "DATABASE_URL": DATABASE_URL,
    "WHATSAPP_TOKEN": WHATSAPP_TOKEN,
    "WHATSAPP_PHONE_ID": WHATSAPP_PHONE_ID,
    "VERIFY_TOKEN": VERIFY_TOKEN,
    "DASHBOARD_URL": DASHBOARD_URL,
    "WEBHOOK_SECRET": WEBHOOK_SECRET,
}

for _name, _value in _required.items():
    if not _value:
        raise EnvironmentError(
            f"\n\nMissing required environment variable: {_name}\n"
            f"Open your .env file, add it, and restart the app.\n"
        )
