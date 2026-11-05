import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_ollama import OllamaEmbeddings

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = UnstructuredLoader(file_path="/home/gregor/Documents/langchain-course-clone/langchain-course/mediumblog1.txt")
    document = loader.load()
    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print("Ingesting...")
    pinecone_vectors = PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))