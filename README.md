# A Simple RAG Serverless Workflow using Free Open Source Tools

This project implements RAG (Retrieval Augmented Generation) workflow using AWS services.

## Architecture

TBD

The workflow consists of three main steps:

### 0️⃣ Document Indexing
- Documents are processed and converted to embeddings using Nomic Embed Text
- The embeddings are stored in locally in a FAISS vector database

### 1️⃣ Query Processing
- User query is converted to an embedding using the same model
- Similar documents are retrieved from the vector index based on embedding similarity
- Retrieved documents become the context for the LLM

### 2️⃣ Response Generation
- OLLAMA generates a response using:
  - The original query
  - The retrieved context documents
- Returns a structured response with:
  - Answer to the query
  - Source documents used for the answer

## Create conda environment
```bash
conda create -n local-rag python=3.10
```

## Install dependencies
```bash
conda activate local-rag
pip install -r requirements.txt
```

For a step-by-step guide on setting up and running OLLAMA on Windows, macOS, and Linux, refer to this guide: [OLLAMA Installation Guide](https://medium.com/@sridevi17j/step-by-step-guide-setting-up-and-running-ollama-in-windows-macos-linux-a00f21164bf3).

## Index documents
```bash
# Make sure you have .txt documents in the documents folder
python document_indexer.py
```

## Query documents
```bash
python query_documents.py
```

Sample output:
```
Enter your query (or 'quit' to exit):
What is the capital of France?

Choose search type:
1. Similarity Search (just find similar documents)
2. QA Search (use LLM to answer question)

1
{
  "answer": "",
  "sources": [
    {
      "content": "...",
      "source": "path/to/document",
      "metadata": {
        "page": 1,
        "other_metadata": "value"
      }
    }
  ]
}

2
{
  "answer": "The LLM's answer to the question",
  "sources": [
    {
      "content": "...",
      "source": "path/to/document",
      "metadata": {
        "page": 1,
        "other_metadata": "value"
      }
    }
  ]
}
```
