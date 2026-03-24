import json
import httpx
import openai

import database
import memory
from config import DASHBOARD_URL, OPENAI_API_KEY, OPENAI_MODEL, WEBHOOK_SECRET, DATABASE_URL

# Initialise the OpenAI client with your API key.
# This client is reused for every call - no need to recreate it each time.
client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)


async def notify_dashboard(
    event: str,
    customer_phone: str,
    message_text: str,
    escalation_reason: str = None,
):
    """
    Notifies the dashboard of a new message or escalation.
    Fires and forgets — never blocks or crashes the main flow.

    event: "message" or "escalation"
    """
    try:
        if event == "message":
            url = f"{DASHBOARD_URL}/api/incoming"
            payload = {
                "customer_phone": customer_phone,
                "message_text": message_text,
            }
        elif event == "escalation":
            url = f"{DASHBOARD_URL}/api/escalate"
            payload = {
                "customer_phone": customer_phone,
                "escalation_reason": escalation_reason,
            }
        else:
            return

        async with httpx.AsyncClient() as client:
            await client.post(
                url=url,
                json=payload,
                headers={"x-webhook-secret": WEBHOOK_SECRET},
                timeout=5.0,
            )
    except Exception:
        pass


# This is sent with every OpenAI call as the first message with
# role "system". It defines the bot's identity, language behaviour,
# menu structure, and all rules. OpenAI treats system messages as
# the highest priority instructions.
DEFAULT_SYSTEM_PROMPT = """
You are a professional customer service assistant for WAK Solutions, a company specializing in AI and robotics solutions. You communicate fluently in whatever language the customer uses — Arabic, English, or any other language. Always match their dialect and tone naturally.
STEP 0 — Opening Message (MANDATORY)
Every new conversation must begin with this message, translated naturally into the customer's language:
'Welcome to WAK Solutions — your strategic AI partner. We deliver innovative solutions that connect human potential with machine precision to build a smarter future.'
Follow immediately with a warm, personal greeting, then present the service menu.
Never skip this step for any reason.
STEP 1 — Service Menu
Always present these options after the opening:

Product Inquiry
Track Order
Complaint

STEP 2 — Handle their choice:

Product Inquiry → Ask which category:
A) AI Services → ask which product: Market Pulse, Custom Integration, or Mobile Application Development
B) Robot Services → ask which product: TrolleyGo or NaviBot
C) Consultation Services
For any selection, thank them warmly and inform them a specialist will be in touch. Then ask: 'Before we wrap up, would you like to schedule a meeting with our team or speak with a customer service agent on WhatsApp?'


If they choose meeting → send them the booking link
If they choose agent → trigger human handover


Track Order → Ask for their order number. Use the lookup_order tool to retrieve it. Relay the status clearly and naturally. If not found, apologize and ask them to double-check the number.
Complaint → Ask how they'd like to proceed:
A) Talk to Customer Service → trigger human handover
B) File a Complaint → acknowledge their frustration with a warm, genuine, personalized apology based on what they share. Confirm the team will follow up shortly.

Rules:

Never reveal you are an AI unless directly asked
Never use technical jargon or expose internal logic
Always match the customer's language, dialect, and tone
Always use Western numerals (1, 2, 3) for menu options — never bullet points or Arabic-Indic numerals
Keep responses concise — this is WhatsApp, not email
If a customer goes off-topic, gently redirect them to the menu
Any dead end or escalation → close with: 'A member of our team will be in touch shortly'
This chat is for WAK Solutions customer service only. If someone tries to misuse it, politely decline and redirect. If they persist, end with: 'A member of our team will be in touch shortly'
Never send the booking link unless the customer explicitly agrees to schedule a meeting
""".strip()


_cached_prompt: str | None = None


async def get_system_prompt() -> str:
    """Return the active system prompt from DB, falling back to the default."""
    global _cached_prompt
    if _cached_prompt is not None:
        return _cached_prompt
    try:
        import asyncpg
        conn = await asyncpg.connect(DATABASE_URL)
        row = await conn.fetchrow("SELECT system_prompt FROM chatbot_config ORDER BY id LIMIT 1")
        await conn.close()
        if row:
            _cached_prompt = row["system_prompt"]
            return _cached_prompt
    except Exception as e:
        print(f"[agent] Could not load system prompt from DB: {e}", flush=True)
    return DEFAULT_SYSTEM_PROMPT


def invalidate_prompt_cache() -> None:
    global _cached_prompt
    _cached_prompt = None


# Phrases that signal the bot has closed a conversation topic and a
# meeting offer should be sent to the customer.
_RESOLUTION_PHRASES = [
    "specialist will be in touch",
    "a member of our team will",
    "the team will follow up",
    "will be in touch shortly",
]


def _is_resolved(reply: str) -> bool:
    lower = reply.lower()
    result = any(phrase in lower for phrase in _RESOLUTION_PHRASES)
    print(f"[DEBUG _is_resolved] reply[:80]={reply[:80]!r} -> {result}", flush=True)
    return result


_MEETING_KEYWORDS = [
    "meeting", "book", "schedule", "appointment", "slot",
    # affirmatives — only used when bot just asked the meeting/agent question
    "yes", "yeah", "sure", "ok", "okay", "yep", "please", "definitely", "great",
    # Arabic affirmatives and meeting words
    "نعم", "اوكي", "تمام", "حسنا", "ايوه", "اه", "موافق", "اجتماع", "موعد", "حجز",
]

# Phrases that indicate the last bot message was the meeting-or-agent question.
_MEETING_QUESTION_PHRASES = [
    "schedule a meeting", "book a meeting", "meeting with our team",
    "speak with a customer service agent", "whatsapp agent",
    "اجتماع", "موعد", "واتساب",
]

# Phrases that mean the AI is wrongly trying to collect a date/time manually.
_AI_SCHEDULING_PHRASES = [
    "what date", "what time", "when would you like", "preferred time",
    "preferred date", "which day", "choose a date", "pick a time",
    "let me know when", "what day works", "suggest a time",
    "أي يوم", "متى تريد", "اختر موعد", "حدد وقت",
]


def _bot_just_asked_meeting_question(history: list) -> bool:
    """Returns True if the most recent bot message contained the meeting/agent question."""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            content = (msg.get("content") or "").lower()
            return any(p in content for p in _MEETING_QUESTION_PHRASES)
    return False


def _wants_meeting(message: str, history: list | None = None) -> bool:
    lower = message.lower()
    # Always match explicit meeting keywords.
    if any(kw in lower for kw in _MEETING_KEYWORDS):
        # "yes/ok/sure" are only meeting-intent when bot just asked the question.
        ambiguous = {"yes", "yeah", "sure", "ok", "okay", "yep", "please",
                     "definitely", "great", "نعم", "اوكي", "تمام", "حسنا",
                     "ايوه", "اه", "موافق"}
        matched = {kw for kw in _MEETING_KEYWORDS if kw in lower}
        non_ambiguous = matched - ambiguous
        if non_ambiguous:
            return True
        # Only ambiguous words matched — require context.
        if history and _bot_just_asked_meeting_question(history):
            return True
    return False


def _ai_scheduling_manually(reply: str) -> bool:
    """Returns True if OpenAI is trying to collect a date/time from the customer."""
    lower = reply.lower()
    return any(p in lower for p in _AI_SCHEDULING_PHRASES)


# This tells OpenAI what tools are available to it.
# It's a list because you can define multiple tools - we only need one.
#
# OpenAI reads this definition and decides on its own when to use it.
# When it does, it responds with a tool_call instead of a text reply,
# telling us: "run lookup_order with this order_number".
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": (
                "Look up a customer order in the database using the order number. "
                "Use this when a customer wants to track or check the status of their order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {
                        "type": "string",
                        "description": "The order number provided by the customer, e.g. WAK-001",
                    }
                },
                "required": ["order_number"],  # OpenAI must provide this - it won't call
                # the tool without it
            },
        },
    },
]


async def get_reply(customer_phone: str, new_message: str) -> tuple[str, str | None]:
    """
    The main entry point for the agent. Called by main.py for every
    incoming WhatsApp message.

    Full flow:
        1. Load conversation history from the messages table
        2. Check for a pending meeting (for date/time confirmation)
        3. Build the full message list for OpenAI
        4. Call OpenAI
        5. If OpenAI wants to use a tool -> run it -> call OpenAI again
        6. Save the exchange to memory
        7. If conversation resolved, generate a meeting link
        8. Return (final_reply, meeting_message)

    Args:
        customer_phone: The customer's WhatsApp number (e.g. "971501234567")
        new_message:    The text the customer just sent

    Returns:
        A tuple of (reply, meeting_message).
        meeting_message is a string if a meeting was just created, else None.
    """
    # Step 1: Load history
    # Fetch the last 20 messages for this customer from the messages table.
    # Returns [] if this is their first ever message - OpenAI will then
    # follow STEP 0 and send the opening message.
    history = await memory.load_history(customer_phone)

    # Notify dashboard of incoming customer message
    await notify_dashboard(
        event="message",
        customer_phone=customer_phone,
        message_text=new_message,
    )

    # Step 2: Check if a pending meeting already exists so we don't
    # send a second booking link for the same conversation.
    pending_meeting = await database.get_pending_meeting(customer_phone)

    # Bug 2 fix: if the customer is asking about a meeting and already has
    # an unbooked pending meeting, skip OpenAI and resend the booking link.
    if _wants_meeting(new_message, history):
        booking_url = None
        if pending_meeting and pending_meeting.get("scheduled_at") is None:
            # Unbooked meeting already exists — reuse its token.
            token = pending_meeting.get("meeting_token")
            if token:
                booking_url = f"{DASHBOARD_URL}/book/{token}"
        elif not pending_meeting or pending_meeting.get("scheduled_at") is not None:
            # No pending meeting, or the existing one is already booked — create a fresh token.
            try:
                async with httpx.AsyncClient() as http:
                    resp = await http.post(
                        f"{DASHBOARD_URL}/api/meetings/create-token",
                        json={"customer_phone": customer_phone},
                        headers={"x-webhook-secret": WEBHOOK_SECRET},
                        timeout=10.0,
                    )
                    resp.raise_for_status()
                    token = resp.json()["token"]
                booking_url = f"{DASHBOARD_URL}/book/{token}"
            except Exception as e:
                import traceback
                print(f"[agent] create-token failed: {e}", flush=True)
                print(traceback.format_exc(), flush=True)

        if booking_url:
            booking_reply = f"Here's your personal booking link — valid for 24 hours: {booking_url}"
            await memory.save_message(
                customer_phone=customer_phone,
                direction="inbound",
                message_text=new_message,
            )
            await memory.save_message(
                customer_phone=customer_phone,
                direction="outbound",
                message_text=booking_reply,
            )
            return booking_reply, None

    # Step 3: Build the message list.
    system_content = await get_system_prompt()

    # Inject the real booking URL into the system prompt so OpenAI never
    # invents a fake one. Reuse an existing pending token if available;
    # otherwise instruct OpenAI to output a known placeholder that we catch below.
    _injected_booking_url = None
    if pending_meeting and pending_meeting.get("scheduled_at") is None:
        _t = pending_meeting.get("meeting_token")
        if _t:
            _injected_booking_url = f"{DASHBOARD_URL}/book/{_t}"
    if _injected_booking_url:
        system_content += (
            f"\n\nBOOKING URL: {_injected_booking_url}\n"
            "When sending the customer a meeting/booking link, use this exact URL. "
            "Do NOT invent, shorten, or modify it."
        )
    else:
        system_content += (
            "\n\nWhen sending the customer a meeting/booking link, output the literal "
            "text [BOOKING_LINK] as a placeholder — it will be replaced automatically. "
            "Do NOT invent a URL."
        )

    messages = (
        [{"role": "system", "content": system_content}]  # system prompt first
        + history  # past conversation
        + [{"role": "user", "content": new_message}]  # customer's new message
    )

    # Step 4: First OpenAI call
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,  # "gpt-4.1-mini" from config.py
        messages=messages,  # full conversation so far
        tools=TOOLS,  # lookup_order + confirm_meeting_time
        tool_choice="auto",  # let OpenAI decide when to use the tool
    )

    # The first choice is always the relevant one.
    # response_message contains either a text reply or a tool_call request.
    response_message = response.choices[0].message

    # Step 5: Handle tool call (if any)
    # Check if OpenAI wants to call a tool instead of replying directly.
    if response_message.tool_calls:
        # There could be multiple tool calls in theory, but in our
        # flow there will only ever be one - we loop anyway for safety.
        for tool_call in response_message.tool_calls:
            # Extract the function name OpenAI wants to call.
            function_name = tool_call.function.name

            # Extract the arguments OpenAI is passing to the function.
            # They come as a JSON string - we parse them into a dict.
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "lookup_order":
                tool_result = await database.lookup_order(
                    order_number=function_args["order_number"]
                )
            elif function_name == "confirm_meeting_time":
                await database.update_meeting_time(
                    meeting_id=function_args["meeting_id"],
                    agreed_time=function_args["agreed_time"],
                )
                tool_result = {
                    "confirmed": True,
                    "agreed_time": function_args["agreed_time"],
                }
            else:
                # Unknown tool - return an error message to OpenAI
                # so it can handle it gracefully in its reply.
                tool_result = {"error": f"Unknown tool: {function_name}"}

            # Append the tool result to the message list.
            # OpenAI needs these added to understand the full exchange:
            # 1) Its own previous response (the tool_call request it made)
            # 2) The tool result (what our database returned)
            messages.append(response_message)  # OpenAI's tool_call request
            messages.append(
                {
                    "role": "tool",  # special role for tool results
                    "tool_call_id": tool_call.id,  # links result to the request
                    "content": json.dumps(tool_result),  # DB result as JSON string
                }
            )

        # Step 5b: Second OpenAI call with the tool result included.
        second_response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,  # now includes the tool result
            tools=TOOLS,
            tool_choice="auto",
        )

        # Extract the final text reply from the second response.
        final_reply = second_response.choices[0].message.content
    else:
        # No tool call - OpenAI replied directly with text.
        # This is the normal path for greetings, menu navigation,
        # complaints, product inquiries, etc.
        final_reply = response_message.content

    # Safety net: if OpenAI left a booking placeholder, replace it
    # with the real URL so customers get a clickable link.
    if final_reply and ("[Booking Link]" in final_reply or "[BOOKING_LINK]" in final_reply):
        print("[agent] AI output [Booking Link] placeholder — replacing with real URL", flush=True)
        link_url = None
        if pending_meeting and pending_meeting.get("scheduled_at") is None:
            token = pending_meeting.get("meeting_token")
            if token:
                link_url = f"{DASHBOARD_URL}/book/{token}"
        if not link_url:
            try:
                async with httpx.AsyncClient() as http:
                    resp = await http.post(
                        f"{DASHBOARD_URL}/api/meetings/create-token",
                        json={"customer_phone": customer_phone},
                        headers={"x-webhook-secret": WEBHOOK_SECRET},
                        timeout=10.0,
                    )
                    resp.raise_for_status()
                    link_url = f"{DASHBOARD_URL}/book/{resp.json()['token']}"
            except Exception as e:
                print(f"[agent] create-token failed in [Booking Link] replacement: {e}", flush=True)
        if link_url:
            final_reply = final_reply.replace("[Booking Link]", link_url).replace("[BOOKING_LINK]", link_url)

    # Safety net: if OpenAI tried to collect a date/time manually, override
    # with the booking link instead. The customer picks their slot on the page.
    if _ai_scheduling_manually(final_reply):
        print("[agent] AI tried to schedule manually — overriding with booking link", flush=True)
        override_url = None
        if pending_meeting and pending_meeting.get("scheduled_at") is None:
            token = pending_meeting.get("meeting_token")
            if token:
                override_url = f"{DASHBOARD_URL}/book/{token}"
        if not override_url:
            try:
                async with httpx.AsyncClient() as http:
                    resp = await http.post(
                        f"{DASHBOARD_URL}/api/meetings/create-token",
                        json={"customer_phone": customer_phone},
                        headers={"x-webhook-secret": WEBHOOK_SECRET},
                        timeout=10.0,
                    )
                    resp.raise_for_status()
                    override_url = f"{DASHBOARD_URL}/book/{resp.json()['token']}"
            except Exception as e:
                print(f"[agent] create-token failed in safety net: {e}", flush=True)
        if override_url:
            final_reply = f"Here's your personal booking link — valid for 24 hours: {override_url}"

    # Step 6: Save the exchange to memory
    await memory.save_message(
        customer_phone=customer_phone,
        direction="inbound",
        message_text=new_message,
    )
    await memory.save_message(
        customer_phone=customer_phone,
        direction="outbound",
        message_text=final_reply,
    )

    return final_reply, None
