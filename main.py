from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()


def main():
    print("Hello from langchain-course!")
    print(os.environ.get("OPENAI_API_KEY"))


if __name__ == "__main__":
    main()
