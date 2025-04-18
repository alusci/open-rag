import json
from utils.vector_store import get_vectorstore
from utils.models import initialize_llm
from utils.response_formatter import format_response
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

def qa_search(query, vectorstore):
    """Perform QA search with OLLAMA LLM"""
    llm = initialize_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant that provides accurate information based on the given context.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer the question based only on the provided context. If the context doesn't contain the relevant information, 
    say "I don't have enough information to answer this question." Don't make up information.
    """)
    
    # Create a document chain that combines documents with the stuffing strategy
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the retrieval chain
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | document_chain
    )
    
    # Run the chain and capture documents separately for source tracking
    retrieved_docs = retriever.invoke(query)
    result = qa_chain.invoke({"question": query})
    
    return format_response(
        answer=result,
        sources=retrieved_docs
    )

def main():
    vectorstore = get_vectorstore()
    
    while True:
        print("\nEnter your query (or 'quit' to exit):")
        query = input()
        
        if query.lower() == 'quit':
            break
            
        print("\nChoose search type:")
        print("1. Similarity Search (just find similar documents)")
        print("2. QA Search (use OLLAMA LLM to answer question)")
        choice = input()
        
        if choice == "1":
            results = vectorstore.similarity_search(query, k=5)
            formatted_response = format_response(sources=results)
        
        elif choice == "2":
            formatted_response = qa_search(query, vectorstore)
        
        else:
            print("Invalid choice. Please choose 1 or 2.")
            continue
        
        print("\nResults:")
        print(json.dumps(formatted_response, indent=2))

if __name__ == "__main__":
    main()