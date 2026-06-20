from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import glob
import os
import uuid


@dataclass
class VectorDBConfig:
    """Configuration for VectorDBManager initialization."""
    
    # Token-based chunking configuration for GPT-4 context window
    MAX_LLM_CONTEXT: int = 128000  # Maximum context window in tokens
    MAIN_CHUNK_SIZE: int = 96000  # 3/4 of budget (96,000 tokens)
    OVERLAP_SIZE: int = 32000     # 1/4 overlap (32,000 tokens)
    
    # ChromaDB settings
    db_path: str = ""
    workspace_path: str = ""
    collection_name: str = "rag_collection"
    max_workers: int = 4
    
    @classmethod
    def from_defaults(cls):
        """Create a config with default values."""
        return cls()
    
    @classmethod
    def from_args(cls, db_path: str, workspace_path: str, collection_name: str = None, max_workers: int = None):
        """Create a config from command-line or user-provided arguments."""
        
        return cls(
            db_path=db_path,
            workspace_path=workspace_path,
            collection_name=collection_name or "rag_collection",
            max_workers=max_workers or 4
        )

class VectorDBManager:
    
    def __init__(self, config: VectorDBConfig):
        # Use the config object for all configuration
        self.db_path = str(Path(config.db_path).resolve())
        
        self.workspace_path = str(Path(config.workspace_path).resolve())
        
        self.collection_name = config.collection_name
        
        self.max_workers = config.max_workers

        self.client: PersistentClient = self._initialize_client()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, embedding_function=DefaultEmbeddingFunction()
        )

        self.MAIN_CHUNK_SIZE = config.MAIN_CHUNK_SIZE
        self.OVERLAP_SIZE = config.OVERLAP_SIZE

        print(f"VectorDBManager initialized at {self.db_path} for {self.collection_name}")
        
        self.sync_workspace()

    def _initialize_client(self) -> PersistentClient:
        """Initialize and return the ChromaDB PersistentClient."""
        # Ensure the directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        return PersistentClient(path=self.db_path)


    def sync_workspace(self):

        """Synchronize the vector DB with the current state of the workspace."""
        
        response = self.collection.get(include=["metadatas"])

        # 1. Changed "source_file" to "source" and added abspath standardization
        current_db_files = {
            os.path.abspath(meta.get("source"))
            for meta in response.get("metadatas", [])
            if meta and "source" in meta and meta.get("source")
        }

        # 2. Extract and standardize all files currently in the workspace folder
        all_entries = glob.glob(os.path.join(self.workspace_path, "*"))
        current_workspace_files = {
            os.path.abspath(p) for p in all_entries if Path(p).is_file()
        }

        # 3. Calculate set differences accurately
        new_files = current_workspace_files - current_db_files
        missing_files = current_db_files - current_workspace_files
        
        # 4. Changed 'elif' to independent 'if' blocks so both run if needed
        if new_files:
            print(f"Syncing: Found {len(new_files)} new file(s) in workspace to ingest.")
            
            # Extract just the filenames (e.g., "doc1.pdf") so DirectoryLoader's glob handles them correctly
            filenames_only = [os.path.basename(f) for f in new_files]
            self._add_files(filenames_only)
        
        if missing_files:
            print(f"Syncing: Missing {len(missing_files)} file(s) from workspace.")
            self._remove_files(missing_files)
        
        if not new_files and not missing_files:
            print("Sync: Workspace is up to date. No changes needed.")


    def _add_files(self, paths: List[str]):
        """
        Ingest new files with retry logic and progress reporting.
        
        Args:
            paths: List of file paths to ingest
            retry_count: Current retry attempt number
            max_retries: Maximum retry attempts
        """
        
        loader = DirectoryLoader(
            path=self.workspace_path,
            glob=paths,
            loader_cls=PyPDFLoader,
            use_multithreading=True,
            max_concurrency=4,
            show_progress=True  
        )

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=self.MAIN_CHUNK_SIZE,
            chunk_overlap=self.OVERLAP_SIZE,
        )

        raw_documents = loader.load()

        file_chunks = text_splitter.split_documents(raw_documents)

        documents_list = []
        metadatas_list = []
        ids_list = []

        for chunk in file_chunks:
            documents_list.append(chunk.page_content)
            metadatas_list.append(chunk.metadata)  # Automatically preserves source & page dictionary
            ids_list.append(f"id_{uuid.uuid4()}")  # Gen unique database primary key string

        if file_chunks:

            self.collection.upsert(
                ids=ids_list,
                documents=documents_list,
                metadatas=metadatas_list
            )
        
            print(f"Successfully loaded {len(file_chunks)} chunks into ChromaDB via LangChain Store.")
    
    def _remove_files(self, file_paths: List[str]) :
        """
        Removes files from the vector DB. Should be called with full file paths.
        
        Uses the delete-by-filter method for O(log n) efficiency.
        Returns the number of files removed.
        
        :param file_paths: List of full file paths to remove.
        :return: Number of files successfully removed from the vector DB.
        """
        if not file_paths:
            print("No file paths provided.")
            return 0
        
        # Convert to set for filter operation
        file_paths_set = list(set(file_paths))
        
        print(f"Deleting {len(file_paths)} entries from collection using filter...")
        
        try:

            self.collection.delete(
                where={"source_file": {"$in": list(file_paths_set)}}
            )
            
            print(f"\n✅ Successfully removed {len(file_paths)} file(s) from vector DB.")
        except Exception as e:
            print(f"Error during deletion: {e}")
    
    def retrieve_context(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        
        """Retrieves relevant documents based on a query.
        
        Returns a list of dictionaries containing 'page_content' and 'metadata' for each retrieved document.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas"]
            )
            # Extract the text content from the resulting documents
            if not results or "documents" not in results:
                return []
            
            # Return list of dicts with page_content and metadata
            return [
                {"page_content": doc, "metadata": meta}
                for doc, meta in zip(results["documents"], results.get("metadatas", [None]))
            ]
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

        if not context_docs or not context_docs[0].get("page_content"):
            print("WARNING: No context retrieved. Proceeding with original query.")
            context_str = "No relevant context found."
        else:
            # Concatenate retrieved documents into a clear context string
            context_str = "\n---\n".join([
                f"Document:\n{doc.get('page_content', '')}" 
                for doc in context_docs
                if doc.get("page_content")
            ])

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