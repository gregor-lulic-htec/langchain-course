from unittest import result

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite",)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 jobs in the bay area on linkedin for an ai engineer using langchain.")})
    print(result)



if __name__ == "__main__":
    main()
