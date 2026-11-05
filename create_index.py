import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
load_dotenv()

# Initialize the native Pinecone client
def run():
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    index_name = os.environ.get("INDEX_NAME")

    # Check if index already exists to avoid conflict errors
    if index_name not in pc.list_indexes().names():
        print(f"Creating a standard vector index: {index_name}...")

        # This provisions a standard vector index (No document schema specified)
        pc.create_index(
            name=index_name,
            dimension=768,          # Change this to match your embedding model (e.g., 1536 for OpenAI)
            metric="cosine",          # Options: "cosine", "euclidean", "dotproduct"
            spec=ServerlessSpec(
                cloud="aws",          # Choose "aws" or "gcp"
                region="us-east-1"    # Match your preferred serverless region
            )
        )
        print("Index created successfully!")
    else:
        print(f"Index '{index_name}' already exists.")


if __name__ == "__main__":
    run()
