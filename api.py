from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import retrieval_qa as RetrievalQA

# Initialize embeddings (use the same model as before)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def rag_api(query: str, llm, k: int = 3, persist_directory: str = "./chroma_db"):
    """
    API-like function to query the RAG system using an LLM.

    Args:
        query (str): The query to search for.
        llm: The LLM instance (e.g., LlamaCpp, HuggingFacePipeline, or any LangChain-compatible LLM).
        k (int): Number of chunks to retrieve. Defaults to 3.
        persist_directory (str): Path to the ChromaDB folder. Defaults to "./chroma_db".

    Returns:
        dict: A dictionary containing:
            - "answer" (str): The generated answer from the LLM.
            - "sources" (list): List of retrieved chunks (with metadata).
    """
    # Load the vector store
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    # Create a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # Create a RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Run the query
    response = qa_chain({"query": query})

    # Format the response
    return {
        "answer": response["result"],
        "sources": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in response["source_documents"]
        ]
    }