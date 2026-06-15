from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from chromadb import PersistentClient, Collection
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from tools import convert_pdf_to_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

class VectorDBManager:
    """
    Manages interactions with a Chroma vector store for RAG operations.
    Handles document ingestion, retrieval, and query augmentation.
    """
    
    def __init__(self, db_path: str, workspace_path: str, collection_name: str = "rag_collection", max_workers: int = 4):
        """
        Initialize the VectorDBManager with database and workspace paths.
        
        Args:
            db_path: Path to the ChromaDB storage directory
            workspace_path: Path to the workspace containing PDF documents
            collection_name: Name of the ChromaDB collection (default: "rag_collection")
            max_workers: Number of concurrent threads for ingestion (default: 4)
        """
        self.db_path = str(Path(db_path).resolve())
        self.workspace_path = str(Path(workspace_path).resolve())
        self.collection_name = collection_name
        self.max_workers = max_workers
        
        self.client: PersistentClient = self._initialize_client()
        self.collection = self.client.get_or_create_collection(
            name = self.collection_name,
            embedding_function=DefaultEmbeddingFunction()
        )
        
        # Track state
        self._last_processed_file_count: int = 0
        
        print(f"VectorDBManager initialized at {self.db_path} for {self.collection_name}")
        self._sync_with_workspace()

    def _initialize_client(self) -> PersistentClient:
        """Initialize and return the ChromaDB PersistentClient."""
        # Ensure the directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        return PersistentClient(path=self.db_path)
    
    def _ingest_file(self, path: str) -> bool:
        """Ingest a single file. Returns True on success, False on failure."""
        try:
            # Read the full text
            full_text = convert_pdf_to_text(path)

            # Split text into manageable chunks
            file_metadata = {"source_file": path}

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            splitted = text_splitter.create_documents([full_text], [file_metadata])

            text_chunks = [splitted[i].page_content for i in range(len(splitted))]

            # Generate IDs using full file path and chunk number
            text_ids = [f"{path}_chunk_{i}" for i in range(1, len(text_chunks) + 1)]

            # Update metadata to include chunk index
            updated_text_metadata = []
            for i in range(len(splitted)):
                meta = splitted[i].metadata.copy()
                meta["chunk_index"] = i
                updated_text_metadata.append(meta)

            assert len(updated_text_metadata) == len(text_ids) == len(text_chunks)

            # Ingest into ChromaDB
            self.collection.upsert(
                ids=text_ids,
                documents=text_chunks,
                metadatas=updated_text_metadata
            )
            return True
        
        except Exception as e:
            print(f"❌ Error processing {path}: {e}")
            return False

    def _sync_with_workspace(self):
        """Synchronize the vector DB with the current state of the workspace."""
        new_files = self._get_new_file_paths()
        missing_files = self._get_deleted_file_paths()
        
        if new_files:
            print(f"Syncing: Found {len(new_files)} new file(s) in workspace to ingest.")
            self._ingest_new_files(new_files)
        elif missing_files :
            print(f"Syncing: Missing {len(missing_files)} file(s) from workspace.")
            self._remove_files(missing_files)
        else:
            print("Sync: No new files found.")
    
    def _get_db_files(self) -> Set[str]:
        """
        Extract file paths from ChromaDB metadata.
        Returns a set of full source_file paths.
        """
        response = self.collection.get(include=["metadatas"])
        return {
            meta.get("source_file")
            for meta in response.get("metadatas", [])
            if meta and "source_file" in meta and meta.get("source_file")
        }
    
    def _get_workspace_files(self) -> Set[str]:
        """Get all files in the workspace directory (full paths)."""
        all_entries = glob.glob(os.path.join(self.workspace_path, "*"))
        return {p for p in all_entries if Path(p).is_file()}

    def _get_new_file_paths(self) -> List[str]:
        """
        Identifies PDF files present on the filesystem that are NOT present in the vector DB metadata.
        Returns a list of file paths to ingest.
        """
        db_files = self._get_db_files()
        current_files = self._get_workspace_files()
        
        if not db_files:
            return list(current_files)
        
        # Find files in workspace but not in the database
        new_files = current_files - db_files
        return list(new_files)
    
    def _get_deleted_file_paths(self) -> List[str]:
        """
        Identifies PDF files that were previously in the vector DB but no longer exist on disk.
        :return: A list of paths to deleted files that need to be removed from the DB.
        """
        db_files = self._get_db_files()
        
        current_files = self._get_workspace_files()
        
        # 2. Find files that exist in DB but NOT on disk
        deleted_files = db_files - current_files
        
        return list(deleted_files)
    
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
            # Delete all entries whose source_file matches any file in file_paths_set
            # This uses ChromaDB's filter index - O(log n) complexity
            self.collection.delete(
                where={"source_file": {"$in": list(file_paths_set)}}
            )
            
            print(f"\n✅ Successfully removed {len(file_paths)} file(s) from vector DB.")
        except Exception as e:
            print(f"Error during deletion: {e}")
    
    def _ingest_new_files(self, paths: List[str], retry_count: int = 0, max_retries: int = 3):
        """
        Ingest new files with retry logic and progress reporting.
        
        Args:
            paths: List of file paths to ingest
            retry_count: Current retry attempt number
            max_retries: Maximum retry attempts
        """
        failed_file_paths: List[str] = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            # Submit all files to thread pool, track by path
            for path in paths:
                future = executor.submit(self._ingest_file, path)
                futures.append((path, future))
            
            # Wait for all to complete, handle partial failures with progress updates
            for path, future in tqdm(futures, total=len(paths), desc="Ingesting"):
                try:
                    success = future.result()
                    
                    if not success:
                        failed_file_paths.append(path)

                except Exception as e:
                    # Extract path from failed future
                    path_to_retry = path
                    print(f"❌ Error processing {path_to_retry}: {e}")
                    failed_file_paths.append(path_to_retry)
            
            # Check if we should retry based on max_retries

            if failed_file_paths :

                if retry_count < max_retries:
                    
                    print(f"Retrying {len(failed_file_paths)} failed file(s) [Attempt {retry_count + 1}/{max_retries + 1}]...")
                    
                    self._ingest_new_files(
                        failed_file_paths, 
                        retry_count=retry_count + 1, 
                        max_retries=max_retries,
                    )
                else:
                    print("\n✅ All retry attempts exhausted. Failed files will be logged.")
                    # Log final failure report
                    for path in failed_file_paths:
                        print(f"⚠️  File {path} will NOT be ingested after {max_retries + 1} attempts.")

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