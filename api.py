from typing import List, Dict, Any
from chromadb import PersistentClient
from tools import convert_pdf_to_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob

class VectorDBManager:
    """
    Manages all interactions with the Chroma vector store, initialized with a specific path.
    This class handles the setup, ingestion, retrieval, and query augmentation logic.
    """
    def __init__(self, db_path: str, workspace_path: str, collection_name: str = "rag_collection"):
        """
        Initializes the Chroma client and collection pointing to a local directory.
        :param db_path: The absolute or relative path to the ChromaDB directory.
        :param collection_name: The name of the collection to use.
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.workspace_path = workspace_path
        self.last_id = -1
        
        self.client = self._initialize_client()
        
        self.collection = self.client.get_or_create_collection(self.collection_name)
     
        print(f"VectorDBManager initialized. DB path: {self.db_path}, Collection: {self.collection_name}")
        
        self._ensure_data_integrity()

    def _initialize_client(self):
        """Initializes Chroma client pointing to the specified local directory."""
        try:
            # Ensure the directory exists
            os.makedirs(self.db_path, exist_ok=True)
            # Initialize PersistentClient pointing to the local path
            return PersistentClient(path=self.db_path)
        except Exception as e:
            print(f"Error initializing ChromaDB client at {self.db_path}: {e}")
            raise
    
    def _ensure_data_integrity(self):

        """Checks for new files and updates the vector DB if new documents are found."""
        
        new_files_paths = self.get_new_file_paths()
        
        if new_files_paths:
            
            print(f"Found {len(new_files_paths)} new file(s) to ingest.")
            
            self._ingest_new_data(new_files_paths)
        else:
            print("No new files found to ingest.")


    def get_new_file_paths(self) -> List[str]:
        """
        Identifies PDF files present on the filesystem that are NOT present in the vector DB metadata.
        :return: A list of paths to new files that need processing.
        """
        # 1. Fetch all source file names stored in ChromaDB metadata
        response = self.collection.get(include=["metadatas"])
        
        db_files = {
            meta.get("source_file")
            for meta in response.get("metadatas", [])
            if meta and "source_file" in meta
        }
        
        self.last_id = len(db_files) - 1 if len(db_files)>0 else 0
        # 2. Find all PDF files in the local workspace
        search_pattern = os.path.join(self.workspace_path, "*")
        current_files = set(glob.glob(search_pattern))

        # 3. Determine the set of files to ingest (local files not in DB)
        new_files = current_files - db_files

        return list(new_files)
    
    def _ingest_new_data(self, paths: List[str]):
        """Reads, chunks, and ingests data from a list of new file paths."""
        for path in paths:
            print(f"Ingesting new file: {path}")
            try:
                # Read the full text
                full_text = convert_pdf_to_text(path)

                # Split text into manageable chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                text_chunks = text_splitter.create_documents([full_text])

                # Prepare metadata for ingestion
                file_metadata = {"source_file": path}
                text_metadata = [file_metadata] * len(text_chunks)
                text_ids = [i for i in range(self.last_id,self.last_id+len(text_chunks))]

                # Ingest into ChromaDB
                self.collection.upsert(
                    ids=text_ids,
                    documents=text_chunks,
                    metadatas=text_metadata
                )
                print(f"Successfully ingested {len(text_chunks)} chunks from {path}.")
            except Exception as e:
                print(f"Failed to ingest data from {path}: {e}")

    def retrieve_context(self, query: str, n_results: int = 5) -> List[str]:
        """Retrieves relevant documents based on a query."""
        print(f"Retrieving context for query: '{query[:50]}...'")
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents"]
            )
            # Extract the text content from the resulting documents    

            return results
        except Exception as e:
            print(f"Error during context retrieval: {e}")
            return []

    
    def get_augmented_query(self, user_query: str) -> str:
        """
        Retrieves relevant context and constructs a context-aware system prompt
        by augmenting the user query with RAG results.
        Implements the context-aware-llm-query-augmentation skill.
        """
        # 1. Retrieve relevant context
        context_docs = self.retrieve_context(user_query, n_results=4)

        if not context_docs:
            print("WARNING: No context retrieved. Proceeding with original query.")
            context_str = "No relevant context found."
        else:
            # Concatenate retrieved documents into a clear context string
            context_str = "\n---\n".join([f"Document:\n{doc}" for doc in context_docs])

        # 2. Construct the final, augmented prompt
        system_prompt = f"""
        You are an expert assistant. Use the following context provided below to answer the user's question accurately.
        
        --- CONTEXT ---
        {context_str}
        --- END CONTEXT ---

        Based ONLY on the context provided above, answer the following user query:
        
        USER QUERY: {user_query}
        """
        return system_prompt