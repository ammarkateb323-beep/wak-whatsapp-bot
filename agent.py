"""
agent.py — OpenAI orchestration for the WAK WhatsApp bot.

Single responsibility: take a customer message, call OpenAI (with optional
tool use), and return the final reply text. All other concerns live in
dedicated modules:

  intent.py       — keyword detection (wants_meeting, wants_escalation, …)
  prompt.py       — system prompt loading + cache
  tools.py        — OpenAI tool schema definitions
  notifications.py — fire-and-forget dashboard notifications
"""

import json
import logging

import httpx
import openai

import database
import memory
from config import DASHBOARD_URL, OPENAI_API_KEY, OPENAI_MODEL, WEBHOOK_SECRET
from intent import ai_scheduling_manually, wants_escalation, wants_meeting
from notifications import mask_phone, notify_dashboard
from prompt import get_system_prompt
from tools import TOOLS

logger = logging.getLogger(__name__)

# Reused for every call — no need to recreate on each request.
client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _resolve_booking_url(
    customer_phone: str,
    pending_meeting: dict | None,
    company_id: int,
) -> str | None:
    """
    Return a booking URL for the customer.

    Reuses an existing unbooked meeting token if one exists, otherwise
    creates a fresh token via the dashboard API. Returns None on failure.
    """
    # Reuse an existing unbooked token if available.
    if pending_meeting and pending_meeting.get("scheduled_at") is None:
        token = pending_meeting.get("meeting_token")
        if token:
            return f"{DASHBOARD_URL}/book/{token}"

    # Create a new meeting token.
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"{DASHBOARD_URL}/api/meetings/create-token",
                json={"customer_phone": customer_phone, "company_id": company_id},
                headers={"x-webhook-secret": WEBHOOK_SECRET},
                timeout=10.0,
            )
            resp.raise_for_status()
            token = resp.json()["token"]
            logger.info(
                "[INFO] [agent] Meeting token created — phone: %s",
                mask_phone(customer_phone),
            )
            return f"{DASHBOARD_URL}/book/{token}"
    except Exception as exc:
        logger.error(
            "[ERROR] [agent] create-token failed — phone: %s, error: %s",
            mask_phone(customer_phone),
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def get_reply(
    customer_phone: str,
    new_message: str,
    *,
    _save_inbound: bool = True,
    company_id: int = 1,
) -> tuple[str, str | None]:
    """
    Main entry point called by main.py for every incoming WhatsApp message.

    Flow:
        1. Load conversation history from the messages table.
        2. Notify dashboard of the inbound message.
        3. Check for escalation intent — notify dashboard immediately if found.
        4. Check for meeting intent — short-circuit with booking URL if found.
        5. Build the OpenAI message list (system + history + new message).
        6. First OpenAI call (with tool_choice="auto").
        7. If OpenAI requests a tool — run it, then make a second OpenAI call.
        8. Safety nets: replace [BOOKING_LINK] placeholder, override manual
           scheduling attempts.
        9. Save the exchange to memory.
       10. Return (final_reply, None).

    Args:
        customer_phone:  The customer's WhatsApp number, e.g. "971501234567".
        new_message:     The text the customer just sent (or Whisper transcription).
        _save_inbound:   If False, skip saving the inbound message here.
                         Set to False when the caller (process_audio_message) has
                         already saved it with richer metadata (media_url, etc.).
        company_id:      The company owning this WhatsApp number.

    Returns:
        (reply_text, meeting_message) — meeting_message is always None now;
        kept for backward compat with main.py callers.
    """
    # ── Step 1: Load history ──────────────────────────────────────────────────
    history = await memory.load_history(customer_phone, company_id)

    # ── Step 2: Notify dashboard (fire-and-forget) ────────────────────────────
    await notify_dashboard(
        event="message",
        customer_phone=customer_phone,
        message_text=new_message,
        company_id=company_id,
    )

    # ── Step 3: Escalation check ──────────────────────────────────────────────
    if wants_escalation(new_message, history):
        logger.info(
            "[INFO] [agent] Escalation intent detected — phone: %s",
            mask_phone(customer_phone),
        )
        await notify_dashboard(
            event="escalation",
            customer_phone=customer_phone,
            message_text=new_message,
            escalation_reason=new_message,
            company_id=company_id,
        )

    # ── Step 4: Meeting intent short-circuit ──────────────────────────────────
    pending_meeting = await database.get_pending_meeting(customer_phone, company_id)

    if wants_meeting(new_message, history):
        booking_url = await _resolve_booking_url(customer_phone, pending_meeting, company_id)
        if booking_url:
            booking_reply = (
                f"Here's your personal booking link — valid for 24 hours: {booking_url}"
            )
            if _save_inbound:
                await memory.save_message(
                    customer_phone=customer_phone,
                    direction="inbound",
                    message_text=new_message,
                    company_id=company_id,
                )
            await memory.save_message(
                customer_phone=customer_phone,
                direction="outbound",
                message_text=booking_reply,
                company_id=company_id,
            )
            logger.info(
                "[INFO] [agent] Booking link sent — phone: %s",
                mask_phone(customer_phone),
            )
            return booking_reply, None

    # ── Step 5: Build message list ────────────────────────────────────────────
    system_content = await get_system_prompt(company_id)

    # Inject a real booking URL into the system prompt so OpenAI never
    # invents a fake one. If no pending token exists, instruct OpenAI to
    # output a known placeholder that we catch and replace below.
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
        [{"role": "system", "content": system_content}]
        + history
        + [{"role": "user", "content": new_message}]
    )

    # ── Step 6: First OpenAI call ─────────────────────────────────────────────
    logger.info(
        "[INFO] [agent] OpenAI request — model: %s, history_len: %d, phone: %s",
        OPENAI_MODEL,
        len(history),
        mask_phone(customer_phone),
    )

    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    usage = response.usage
    logger.info(
        "[INFO] [agent] OpenAI response — model: %s, prompt_tokens: %s, completion_tokens: %s, phone: %s",
        OPENAI_MODEL,
        usage.prompt_tokens if usage else "n/a",
        usage.completion_tokens if usage else "n/a",
        mask_phone(customer_phone),
    )

    response_message = response.choices[0].message

    # ── Step 7: Tool call handling ────────────────────────────────────────────
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            logger.info(
                "[INFO] [agent] Tool call — function: %s, phone: %s",
                function_name,
                mask_phone(customer_phone),
            )

            if function_name == "lookup_order":
                tool_result = await database.lookup_order(
                    order_number=function_args["order_number"],
                    company_id=company_id,
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
                logger.warning(
                    "[WARN] [agent] Unknown tool requested — function: %s", function_name
                )
                tool_result = {"error": f"Unknown tool: {function_name}"}

            messages.append(response_message)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result),
                }
            )

        # Second OpenAI call with the tool result included.
        second_response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        second_usage = second_response.usage
        logger.info(
            "[INFO] [agent] OpenAI tool-follow-up — model: %s, prompt_tokens: %s, completion_tokens: %s, phone: %s",
            OPENAI_MODEL,
            second_usage.prompt_tokens if second_usage else "n/a",
            second_usage.completion_tokens if second_usage else "n/a",
            mask_phone(customer_phone),
        )
        final_reply = second_response.choices[0].message.content
    else:
        final_reply = response_message.content

    # ── Step 8: Safety nets ───────────────────────────────────────────────────

    # Replace [BOOKING_LINK] placeholder if OpenAI output it.
    if final_reply and (
        "[Booking Link]" in final_reply or "[BOOKING_LINK]" in final_reply
    ):
        logger.info(
            "[INFO] [agent] Replacing [BOOKING_LINK] placeholder — phone: %s",
            mask_phone(customer_phone),
        )
        link_url = await _resolve_booking_url(customer_phone, pending_meeting, company_id)
        if link_url:
            final_reply = final_reply.replace("[Booking Link]", link_url).replace(
                "[BOOKING_LINK]", link_url
            )

    # Override if OpenAI tried to manually collect a date/time.
    if ai_scheduling_manually(final_reply):
        logger.info(
            "[INFO] [agent] Overriding manual scheduling attempt — phone: %s",
            mask_phone(customer_phone),
        )
        override_url = await _resolve_booking_url(customer_phone, pending_meeting, company_id)
        if override_url:
            final_reply = (
                f"Here's your personal booking link — valid for 24 hours: {override_url}"
            )

    # ── Step 9: Save exchange ─────────────────────────────────────────────────
    if _save_inbound:
        await memory.save_message(
            customer_phone=customer_phone,
            direction="inbound",
            message_text=new_message,
            company_id=company_id,
        )
    await memory.save_message(
        customer_phone=customer_phone,
        direction="outbound",
        message_text=final_reply,
        company_id=company_id,
    )

    return final_reply, None
