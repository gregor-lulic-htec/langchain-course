import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
llm = ChatOllama(model="qwen3:1.7b")

vector_store = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question:
{question}
Provide a detailed answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query:str):
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content

def create_retrieval_chain_with_lcel():
    retrieval_chain = (
            RunnablePassthrough.assign(context=itemgetter('question') | retriever | format_docs) | prompt_template | llm | StrOutputParser())
    return retrieval_chain


if __name__ == "__main__":
    query = "What is pinecone in machine learning?"

    print("IMPLEMENTATION 0: Raw llm implementation no RAG")
    print("="*60)
    raw_result = llm.invoke([HumanMessage(content=query)])
    print("\n\nAnswer:")
    print(raw_result.content)
    print("IMPLEMENTATION 1: Implementation without LCEL")
    print("="*60)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\n\nAnswer:")
    print(result_without_lcel)

    print("IMPLEMENTATION 2: Implementation with LCEL")
    print("="*60)
    result_with_lcel = create_retrieval_chain_with_lcel().invoke({"question": query})
    print("\n\nAnswer:")
    print(result_with_lcel)