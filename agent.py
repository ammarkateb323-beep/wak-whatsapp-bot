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
You are a professional customer service assistant for WAK Solutions, a company specializing in AI and robotics solutions. You communicate in whatever language the customer uses - Arabic, English, Chinese, or any other language.

STEP 0 - Opening Message
This step is mandatory and must always be sent as the first message in every new conversation, without exception. Do not skip it for any reason.
Always begin every new conversation with this message, translated naturally into the customer's language:
"Welcome to WAK Solutions - your strategic AI partner. We deliver innovative solutions that connect human potential with machine precision to build a smarter future."
Follow it immediately with a warm personal greeting, then present the service menu.

STEP 1 - Service Menu
After the opening, always present these options:
1. Product Inquiry
2. Track Order
3. Complaint

STEP 2 - Based on their choice:

1. Product Inquiry -> Ask which category:
   A) AI Services -> then ask which product: Market Pulse, Custom Integration, or Mobile Application Development
   B) Robot Services -> then ask which product: TrolleyGo or NaviBot
   C) Consultation Services
   For any product or consultation selection, thank them warmly and let them know a specialist will be in touch. End the conversation politely.

2. Track Order -> Ask them to share their order number. Use the lookup_order tool to look up the order by order_number. Relay the status and details naturally and clearly. If no order is found, apologize and suggest they double-check the number.

3. Complaint -> Ask how they'd like to proceed:
   A) Talk to Customer Service -> tell them a team member will be with them shortly
   B) File a Complaint -> acknowledge their frustration with a warm, genuine, personalised apology based on what they share. Let them know the team will follow up.

Rules:
- Never mention you are an AI unless directly asked
- Never use technical jargon or show internal logic
- Always match the customer's language and tone
- Always present menu options as numbered lists using Western numerals (1, 2, 3) regardless of language - never use bullet points or Arabic-indic numerals
- Keep responses concise - this is WhatsApp, not email
- If a customer goes off-topic, gently redirect them to the menu
- Any dead end or escalation -> politely close with "A member of our team will be in touch shortly"
- This WhatsApp chat is for WAK Solutions customer service only. If a customer requests unrelated help, politely decline and redirect them to the menu. If they repeatedly try to misuse the chat, end the conversation politely with "A member of our team will be in touch shortly"
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


_MEETING_KEYWORDS = ["meeting", "book", "schedule", "appointment", "slot", "call"]


def _wants_meeting(message: str) -> bool:
    lower = message.lower()
    return any(kw in lower for kw in _MEETING_KEYWORDS)


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
    {
        "type": "function",
        "function": {
            "name": "confirm_meeting_time",
            "description": (
                "Confirm and save the date and time the customer has chosen for their video meeting. "
                "Call this when the customer replies with their preferred date and time for the meeting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "meeting_id": {
                        "type": "integer",
                        "description": "The ID of the pending meeting to update.",
                    },
                    "agreed_time": {
                        "type": "string",
                        "description": "The date and time the customer specified, e.g. 'Monday 10 AM', 'March 20 at 3 PM'.",
                    },
                },
                "required": ["meeting_id", "agreed_time"],
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
    if (
        _wants_meeting(new_message)
        and pending_meeting
        and pending_meeting.get("scheduled_at") is None
    ):
        token = pending_meeting.get("meeting_token")
        if token:
            booking_url = f"{DASHBOARD_URL}/book/{token}"
            booking_reply = (
                f"You can book your preferred time slot here: {booking_url}"
            )
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

    # Step 6: Save the exchange to memory
    # Save both the customer's message and the bot's reply to the
    # messages table so future calls have accurate history.
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

    # Step 7: If the conversation was just resolved, send the customer a
    # booking link. If an unbooked pending meeting already exists, resend
    # its token (Bug 1 fix). Otherwise create a fresh one.
    meeting_message: str | None = None
    print(f"[DEBUG step7] _is_resolved={_is_resolved(final_reply)} pending_meeting={pending_meeting}", flush=True)
    if _is_resolved(final_reply):
        if pending_meeting and pending_meeting.get("scheduled_at") is None:
            # Unbooked meeting already exists — resend the existing booking URL.
            token = pending_meeting.get("meeting_token")
            print(f"[DEBUG step7] Resending existing token={token}", flush=True)
            if token:
                booking_url = f"{DASHBOARD_URL}/book/{token}"
                meeting_message = (
                    f"Thank you for contacting WAK Solutions! Would you like to schedule a "
                    f"video meeting with one of our team? Book your preferred time here: "
                    f"{booking_url}"
                )
        elif not pending_meeting:
            # No meeting yet — create a new booking token.
            print(f"[DEBUG step7] No pending meeting — calling create_meeting_with_token", flush=True)
            try:
                token = await database.create_meeting_with_token(customer_phone)
                print(f"[DEBUG step7] create_meeting_with_token returned token={token}", flush=True)
                booking_url = f"{DASHBOARD_URL}/book/{token}"
                meeting_message = (
                    f"Thank you for contacting WAK Solutions! Would you like to schedule a "
                    f"video meeting with one of our team? Book your preferred time here: "
                    f"{booking_url}"
                )
            except Exception as e:
                import traceback
                print(f"[DEBUG step7] create_meeting_with_token FAILED: {e}", flush=True)
                print(traceback.format_exc(), flush=True)
        else:
            print(f"[DEBUG step7] Pending meeting already booked (scheduled_at={pending_meeting.get('scheduled_at')}) — skipping", flush=True)
    print(f"[DEBUG step7] meeting_message={meeting_message!r}", flush=True)

    return final_reply, meeting_message
