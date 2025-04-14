from utils.document_processor import load_and_split_documents, index_documents
from utils.vector_store import save_vectorstore

def main():
    # Load and split documents
    documents = load_and_split_documents("./documents")
    
    # Index documents
    vectorstore = index_documents(documents)
    save_vectorstore(vectorstore)
    print(f"Successfully indexed {len(documents)} document chunks")

if __name__ == "__main__":
    main() 