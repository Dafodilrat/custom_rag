import os
from api import VectorDBManager
import logging
from openai import OpenAI

# --- Configuration ---
# Configuration settings should ideally be loaded from environment variables or a config file.
PDF_FOLDER_PATH = "./pdf"  # Path to the folder containing PDF documents
VECTOR_DB_PATH = "./chroma_db" # Path where the ChromaDB store will be saved

def call_llm_api(prompt: str) -> str:
    """
    Mocks the call to an external Large Language Model API (Ollama/OpenAI).
    This function handles the LLM interface.
    """
    try :
        client = OpenAI(
            base_url='http://localhost:8000/v1/',
            api_key='ollama'  # Value is required but safely ignored locally
        )
    except Exception as e:
        logging.error(f"An error occurred during LLM client initialization: {e}")
        return "LLM_CLIENT_ERROR"
    
    try:
        response = client.chat.completions.create(
            model='gemma-4-E2B-it-GGUF',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"An error occurred during LLM API call: {e}")
        return "LLM_API_ERROR"

def main():
    """
    Executes the full RAG workflow:
    1. Initialize VectorDBManager.
    2. Chunk documents from the PDF folder.
    3. Ingest chunks into the VectorDB.
    4. Perform a query using the context-aware LLM flow.
    """
    print("--- Starting RAG Workflow ---")

    # 2. Initialize the VectorDB store
    try:
        print(f"Initializing VectorDB at: {VECTOR_DB_PATH}")
        vector_db_manager = VectorDBManager(db_path=VECTOR_DB_PATH,workspace_path=PDF_FOLDER_PATH)

    except Exception as e:
        print(f"Error initializing VectorDBManager: {e}")
        return

    # 5. Call the LLM with a modified query (Context-aware flow)
    print("Executing context-aware query augmentation flow...")

    query = "what is the image and video computing project about?"
    augmented_query = vector_db_manager.get_augmented_query(query)

    # response = call_llm_api(augmented_query)

    # print("\n--- Final LLM Response ---")
    # print(response)


    # print("\n--- RAG Workflow Finished ---")

if __name__ == "__main__":
    main()