import requests
from typing import Dict, Any

# --- LLM Connection Configuration ---
# This section is configured to connect to a local LLM endpoint.
# Ensure the LLM server (like the one running in the Odysseus webui) is running on this host/port.
LLM_HOST = "localhost:7000"
LLM_MODEL = "gemma-4-E2B-it-GGUF"

def call_llm(prompt: str) -> str:
    """
    Calls the local LLM endpoint with the given prompt.
    """
    url = f"{LLM_HOST}/generate"  # Assuming the endpoint path for generation is '/generate'
    payload = {"prompt": prompt}

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        # Assuming the response structure contains the generated text in a specific key
        # NOTE: You may need to adjust this parsing based on the actual response format from your server.
        result = response.json()
        return result.get("generated_text", "Error: Could not find 'generated_text' in the response.")
        
    except requests.exceptions.RequestException as e:
        return f"Error connecting to the LLM at {LLM_HOST}: {e}"


def augment_query(llm_client, original_query: str, context: str) -> str:
    """
    Uses the LLM to rewrite the original query, incorporating retrieved context.
    """
    augmentation_prompt = f"""
    You are an expert query augmentation assistant. Your task is to take an original user query and enrich it by incorporating relevant context provided below. 
    The goal is to create a new, highly specific search query that is more likely to yield a precise answer from a knowledge base.

    Original User Query: "{original_query}"

    Relevant Context Snippets (use these to refine the query):
    ---
    {context}
    ---

    Please return ONLY the new, augmented search query. Do not include any explanation or preamble.
    """
    
    print("--- Augmenting Query via LLM ---")
    augmented_query = call_llm(augmentation_prompt)
    print(f"Augmented Query: {augmented_query}")
    return augmented_query


def rag_api(user_query: str) -> Dict[str, Any]:
    """
    The main RAG API function that handles the full flow:
    1. Retrieval from ChromaDB.
    2. Query Augmentation via LLM.
    3. Final Answer Generation.
    """
    print(f"Received User Query: '{user_query}'")

    # 1. Retrieval
    # In a real setup, you would load your ChromaDB here.
    # For this example, we simulate retrieval:
    retrieved_context = f"Context retrieved for query '{user_query}'. This context is based on the documents found in the ChromaDB."
    
    # 2. Augmentation
    augmented_q = augment_query(
        llm_client=None,  # llm_client is not strictly needed here as call_llm is defined globally
        original_query=user_query,
        context=retrieved_context
    )

    # 3. Final Answer Generation
    final_answer = f"Based on the augmented query ('{augmented_q}'), here is the final answer: {retrieved_context}. Please review the augmented query for precision."

    return {
        "original_query": user_query,
        "augmented_query": augmented_q,
        "context_used": len(retrieved_context),
        "answer": final_answer
    }

if __name__ == '__main__':
    # Example usage for testing
    test_query = "What is the capital of France?"
    result = rag_api(test_query)
    print("\n--- Final RAG Result ---")
    print(f"Original Query: {result['original_query']}")
    print(f"Augmented Query: {result['augmented_query']}")
    print(f"Context Used: {result['context_used']}")
    print(f"Final Answer: {result['answer']}")