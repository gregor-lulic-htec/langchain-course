import sys
from idlelib import __main__
from unittest import result

from annotated_types import BaseMetadata
from dotenv import load_dotenv
import ollama
from langsmith import traceable

load_dotenv()


MODEL = "qwen3:1.7b"
MAX_ITERATIONS = 10


@traceable(run_type="tool")
def get_product_price(product: str):
    """
    Look up the product price
    """

    print(f" >> Executing get_product_price for {product}")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount tier to a price abd return the final price
    Available tiers are bronze, silver, gold"""

    print(f" >> Executing apply_discount for {discount_tier} tier, price: {price}")
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools_for_llm = [
    {
      "type": "function",
      "function": {
        "name": "get_product_price",
        "description": "Look up the product price in the catalog.",
        "parameters": {
          "type": "object",
          "required": ["product"],
          "properties": {
            "product": {"type": "string", "description": "The name of the product, e.g 'laptop', 'headphones', 'keyboard'"}
          }
        }
      }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply discount tier to a price abd return the final price",
            "parameters": {
                "type": "object",
                "required": ["price", "discount_tier"],
                "properties": {
                    "price": {"type": "number", "description": "The price to which to apply discount to"},
                    "discount_tier": {"type": "string", "description": "Tiers with different discount values. e.g: bronze, silver, gold"}
                }
            }
        }
    },
  ]
# ---- Traceble chat ----


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# ---- Agent Loop ----


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print("Question:", question)
    print("=" * 60)

    messages = [
        {"role": "system", "content": """
        You are a helpful shopping assistant,
        You have access to a product catalog tool and a discount tool.
        STRICT RULES - you must follow these exactly: \n
        1. NEVER guess any product price
        You MUST call get_product_price first to get the real price. \n
        2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price 
        returned by get_product_price - DO NOT pass a made-up number. \n
        3. NEVER calculate discounts yourself using math,
        Always use the apply_discount tool. \n
        4. If a user does not specify a discount tier,
        ask them which tier to use - do NOT assume one.
        """},
        {"role": "user", "content": question}
    ]

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n --- Iteration {i} of {MAX_ITERATIONS} ---")

        response = ollama_chat_traced(messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content} ")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        tool_call_id = tool_call.get("id")

        print(f"\nTool selected: {tool_name} with args: {tool_args} ")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use(**tool_args)
        print(f"Tool result: {observation}")


        messages.append(ai_message)
        messages.append({
            "role": "tool",
            "content": str(observation),
        })

    print("ERROR max iterations")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent(
        "What is the price of a laptop after applying a GOLD discount tier?"
    )
