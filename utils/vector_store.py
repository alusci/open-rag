from langchain_community.vectorstores import FAISS
from config import VECTOR_DIMENSION
from .models import initialize_embeddings
import os


def get_vectorstore(documents: list = None):
    """Get or create a vector store instance"""
    
    vectorstore = None
    embeddings = initialize_embeddings()
        
    # Check if index directory exists
    if os.path.exists("./faiss_index"):
        try:
            # Try to load existing index
            vectorstore = FAISS.load_local(
                "./faiss_index", 
                embeddings,
                allow_dangerous_deserialization=True # Don't do this in production!
            )
        except RuntimeError as e:
            print(f"Could not load existing index: {e}")
    
    return vectorstore

def init_vectorstore(documents):
    """Init FAISS vector store"""
    
    embeddings = initialize_embeddings()
    
    return FAISS.from_documents(documents, embeddings, dimension=VECTOR_DIMENSION)

def save_vectorstore(vectorstore):
    """Save the vector store to disk"""
    if vectorstore is not None:
        # Create directory if it doesn't exist
        os.makedirs("./faiss_index", exist_ok=True)
        vectorstore.save_local("./faiss_index") 
