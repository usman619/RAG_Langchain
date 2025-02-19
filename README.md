# RAG Langchain
This project implements a RAG (Retrieval Augmented Generation) system using LangChain with the following main components:

## Database Creation (create_database.py)
- Loads markdown documents from the book directory
- Splits text into smaller chunks (300 characters with 100 character overlap) 
- Creates embeddings using Ollama's nomic-embed-text model
- Stores the embeddings in a Chroma vector database

## Embedding Comparison (compare_embedding.py)
- Generates embeddings for word pairs using Ollama
- Calculates cosine similarity and distance between the embeddings
- Currently compares "apple" and "iphone" as an example
- Can optionally use OpenAI embeddings (commented out)

## Project Structure
- `chroma/` - Vector database storage
- `data/book/` - Source documents directory
- `create_database.py` - Database creation script
- `compare_embedding.py` - Embedding comparison script

## Requirements
- Python 3.8+
- LangChain
- Ollama
- ChromaDB