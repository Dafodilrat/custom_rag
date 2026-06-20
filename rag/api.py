from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
from database import VectorDBManager, VectorDBConfig
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG API",
    description="API for RAG system operations",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global database instance (will be initialized on first request)
_db_instance: VectorDBManager = None


class ResyncRequest(BaseModel):
    """Request model for resync workspace."""
    pass


class LLMConfig(BaseModel):
    """Configuration for LLM connection."""
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000


class QueryRequest(BaseModel):
    """Request model for querying the database."""
    query: str
    n_results: int = 5


class RetrieveRequest(BaseModel):
    """Request model for retrieving content."""
    query: str
    n_results: int = 5


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG API is running",
        "endpoints": {
            "GET /resync": "Resync workspace with database",
            "POST /query": "Get augmented query for LLM",
            "POST /retrieve": "Retrieve context from database"
        }
    }


@app.get("/resync", response_model=Dict[str, Any])
async def resync_workspace(request: ResyncRequest = None):
    """
    Resync the workspace with the vector database.
    
    This endpoint synchronizes all files in the workspace directory with the 
    vector database, adding new files and removing deleted ones.
    """
    global _db_instance
    
    if _db_instance is None:
        _db_instance = _initialize_database()
    
    try:
        # The database manager's __init__ method calls sync_workspace() automatically
        # For explicit control, we can call it here as well
        _db_instance.sync_workspace()
        
        return {
            "status": "success",
            "message": "Workspace resynced successfully",
            "note": "This may take a few minutes depending on the number of files"
        }
    except Exception as e:
        logger.error(f"Error during resync: {e}")
        raise HTTPException(status_code=500, detail=f"Resync failed: {str(e)}")


@app.post("/query", response_model=Dict[str, Any])
async def get_augmented_query(request: QueryRequest):
    """
    Get an augmented query for LLM by retrieving relevant context.
    
    This endpoint retrieves relevant documents from the database and constructs 
    a context-aware system prompt that can be used with an LLM.
    """
    global _db_instance
    
    if _db_instance is None:
        _db_instance = _initialize_database()
    
    try:
        system_prompt = _db_instance.get_augmented_query(request.query)
        
        return {
            "status": "success",
            "system_prompt": system_prompt
        }
    except Exception as e:
        logger.error(f"Error during query augmentation: {e}")
        raise HTTPException(status_code=500, detail=f"Query augmentation failed: {str(e)}")


@app.post("/retrieve", response_model=Dict[str, Any])
async def retrieve_content(request: RetrieveRequest):
    """
    Retrieve relevant context from the database.
    
    This endpoint queries the vector database for documents relevant to the 
    given query and returns them along with their metadata.
    """
    global _db_instance
    
    if _db_instance is None:
        _db_instance = _initialize_database()
    
    try:
        context_str, paths = _db_instance.retrieve_context(request.query, n_results=request.n_results)
        
        return {
            "status": "success",
            "context": context_str,
            "sources": list(paths),
            "query": request.query
        }
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


def _initialize_database() -> VectorDBManager:
    """Initialize the database manager with default configuration."""
    config = VectorDBConfig.from_defaults()
    return VectorDBManager(config)
