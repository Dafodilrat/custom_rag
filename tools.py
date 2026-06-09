import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- Configuration ---
PDF_FOLDER = "pdf"
VECTOR_DB_PATH = "vectordb"
EMBEDDING_MODEL = "nomic-embed-text" # A suitable embedding model for Ollama

def pdf_to_text(pdf_path: str) -> str:
    """
    Loads a single PDF file and extracts all its text content.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text = "\n\n".join([page.page_content for page in pages])
        print(f"Successfully extracted text from {os.path.basename(pdf_path)}.")
        return text
    except Exception as e:
        print(f"Error reading or loading PDF {pdf_path}: {e}")
        return ""

def sentence_based_chunking(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Splits large text into smaller, meaningful chunks based on recursive character splitting.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents([Document(page_content=text)])
    print(f"Text split into {len(chunks)} chunks.")
    return chunks

def load_and_vectorize_data():
    """
    Scans the PDF folder, loads all PDFs, splits them into chunks,
    creates embeddings, and stores them in the Chroma vector database.
    This function is intended to be called by the autoload feature in main.py.
    """
    print(f"--- Starting data loading and vectorization from {PDF_FOLDER} ---")

    # 1. Find all PDF files
    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in the '{PDF_FOLDER}' directory. Nothing to load.")
        return

    documents = []
    
    # 2. Load documents
    for pdf_path in pdf_files:
        print(f"Loading document: {os.path.basename(pdf_path)}")
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {e}")

    if not documents:
        print("No documents were successfully loaded.")
        return

    # 3. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Successfully split {len(documents)} pages into {len(chunks)} chunks.")

    # 4. Initialize Embeddings and Vector Store
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        # Create or load the Chroma vector store
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        print(f"Successfully created and persisted vector database at '{VECTOR_DB_PATH}'")
        print("--- Data loading and vectorization complete! ---")

    except Exception as e:
        print(f"Error during embedding or vector store creation: {e}")
        print("Ensure Ollama is running and the embedding model is available.")


if __name__ == '__main__':
    print("tools.py loaded. Run load_and_vectorize_data() to populate the vector DB.")