import json
import openai

import database
import memory
from config import OPENAI_API_KEY, OPENAI_MODEL

# Initialise the OpenAI client with your API key.
# This client is reused for every call - no need to recreate it each time.
client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)


# This is sent with every OpenAI call as the first message with
# role "system". It defines the bot's identity, language behaviour,
# menu structure, and all rules. OpenAI treats system messages as
# the highest priority instructions.
SYSTEM_PROMPT = """
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
    }
]


async def get_reply(customer_phone: str, new_message: str) -> str:
    """
    The main entry point for the agent. Called by main.py for every
    incoming WhatsApp message.

    Full flow:
        1. Load conversation history from the messages table
        2. Build the full message list for OpenAI
        3. Call OpenAI
        4. If OpenAI wants to use a tool -> run it -> call OpenAI again
        5. Return the final text reply

    Args:
        customer_phone: The customer's WhatsApp number (e.g. "971501234567")
        new_message:    The text the customer just sent

    Returns:
        The bot's reply as a plain string, ready to send via WhatsApp.
    """
    # Step 1: Load history
    # Fetch the last 20 messages for this customer from the messages table.
    # Returns [] if this is their first ever message - OpenAI will then
    # follow STEP 0 and send the opening message.
    history = await memory.load_history(customer_phone)

    # Step 2: Build the message list
    # The system prompt always comes first - it frames everything after it.
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]  # system prompt first
        + history  # past conversation
        + [{"role": "user", "content": new_message}]  # customer's new message
    )

    # Step 3: First OpenAI call
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,  # "gpt-4.1-mini" from config.py
        messages=messages,  # full conversation so far
        tools=TOOLS,  # the lookup_order tool definition
        tool_choice="auto",  # let OpenAI decide when to use the tool
    )

    # The first choice is always the relevant one.
    # response_message contains either a text reply or a tool_call request.
    response_message = response.choices[0].message

    # Step 4: Handle tool call (if any)
    # Check if OpenAI wants to call a tool instead of replying directly.
    # This happens when the customer provides an order number to track.
    if response_message.tool_calls:
        # There could be multiple tool calls in theory, but in our
        # flow there will only ever be one - we loop anyway for safety.
        for tool_call in response_message.tool_calls:
            # Extract the function name OpenAI wants to call.
            function_name = tool_call.function.name

            # Extract the arguments OpenAI is passing to the function.
            # They come as a JSON string - we parse them into a dict.
            function_args = json.loads(tool_call.function.arguments)

            # Run the actual database lookup.
            # We only have one tool, but we use an if-check here so
            # it's easy to add more tools later without restructuring.
            if function_name == "lookup_order":
                tool_result = await database.lookup_order(
                    order_number=function_args["order_number"]
                )
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

        # Step 5: Second OpenAI call
        # Now call OpenAI again with the tool result included.
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
        role="user",
        message_text=new_message,
    )
    await memory.save_message(
        customer_phone=customer_phone,
        direction="outbound",
        role="assistant",
        message_text=final_reply,
    )

    return final_reply
