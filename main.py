from tools import load_and_vectorize_data

from langchain_community.vectorstores import Chroma
from langchain_community.docstore.document import Document

# --- Configuration ---
PDF_FOLDER = "pdf/"
TEXT_FOLDER = "text/"
PERSIST_DIRECTORY = "./chroma_db"
# ---------------------

def setup_vector_store():
    """
    Initializes the vector store by loading and vectorizing all documents.
    """
    vector_store, embeddings = load_and_vectorize_data(
        pdf_folder=PDF_FOLDER,
        text_folder=TEXT_FOLDER,
        persist_directory=PERSIST_DIRECTORY
    )
    return vector_store, embeddings

if __name__ == "__main__":
    print("Starting RAG system setup...")
    try:
        vector_store, embeddings = setup_vector_store()
        print("RAG system setup complete. Vector store is ready.")
    except Exception as e:
        print(f"An error occurred during setup: {e}")