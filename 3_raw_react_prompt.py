import inspect
import re
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
def apply_discount(price: str, discount_tier: str) -> float:
    """Apply discount tier to a price abd return the final price
    Available tiers are bronze, silver, gold"""

    print(f" >> Executing apply_discount for {discount_tier} tier, price: {price}")
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }
    discount = discount_percentages.get(discount_tier, 0)
    return round(float(price) * (1 - discount / 100), 2)


tools = {
    'get_product_price': get_product_price,
    'apply_discount': apply_discount,
}


def get_tool_descriptions(tools_dict):
    descriptions = []

    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, '__wrapped__', tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function)
        descriptions.append(f"{tool_name}{signature} - {docstring}")

    return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())
react_prompt = f"""
 STRICT RULES - you must follow these exactly: \n
        1. NEVER guess any product price
        You MUST call get_product_price first to get the real price. \n
        2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price 
        returned by get_product_price - DO NOT pass a made-up number. \n
        3. NEVER calculate discounts yourself using math,
        Always use the apply_discount tool. \n
        4. If a user does not specify a discount tier,
        ask them which tier to use - do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:
"""
# ---- Traceble chat ----


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


# ---- Agent Loop ----


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    print("Question:", question)
    print("=" * 60)

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n --- Iteration {i} of {MAX_ITERATIONS} ---")

        full_prompt = prompt + scratchpad
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={
                'stop': ['\nObservation'],
                "temperature": 0,
            }
        )
        output = response.message.content
        print("  [Parsing] Looking for the final answer...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("\n" + "="*60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_input_match or not action_match:
            print("\n [PArsing] Action input or action could not be found!")
            break
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"   [Tool Selected] {tool_name} with args: {tool_input_raw}")
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not recognized. Available tools: {tools.keys()}"
        else:
            observation = str(tools[tool_name](*args))

        print(f"   [Tool Result] {observation}]")

        scratchpad += f"{output}\nObservation: {observation}\nThought:"

        # messages.append(ai_message)
        # messages.append({
        #     "role": "tool",
        #     "content": str(observation),
        # })

    print("ERROR max iterations")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent(
        "What is the price of a laptop after applying a GOLD discount tier?"
    )
