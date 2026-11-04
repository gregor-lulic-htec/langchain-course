import sys
from idlelib import __main__
from unittest import result

from annotated_types import BaseMetadata
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langsmith import traceable

load_dotenv()


MODEL = "qwen3:1.7b"
MAX_ITERATIONS = 10


@tool
def get_product_price(product: str):
    """
    Look up the product price
    """

    print(f" >> Executing get_product_price for {product}")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@tool
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


# ---- Agent Loop ----


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {elem.name: elem for elem in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print("Question:", question)
    print("=" * 60)

    messages: list[BaseMessage] = [
        SystemMessage("""
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
        """),
        HumanMessage(question),
    ]

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n --- Iteration {i} of {MAX_ITERATIONS} ---")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content} ")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"\nTool selected: {tool_name} with args: {tool_args} ")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)
        print(f"Tool result: {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR max iterations")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent(
        "What is the price of a laptop after applying a GOLD discount tier?"
    )
