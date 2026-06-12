import os
import sys

# Add the project root to the path so we can import modules from the workspace
sys.path.append("/app/Documents/projects/rag")

try:
    # Attempt to import core modules to check if the environment is set up
    from tools import load_and_vectorize_data
    from api import check_and_load_data

    print("--- Test Initialization Successful ---")
    print("Successfully imported 'tools' and 'api' modules.")
    print("Core functions are accessible.")

    # Simple test call to check API availability
    print("\n--- Testing API Endpoint Call ---")
    result = check_and_load_data()
    print(f"API Call Result: {result}")

except ImportError as e:
    print(f"ERROR: Failed to import a required module. Please check your file structure. Error: {e}")
except Exception as e:
    print(f"ERROR: An unexpected error occurred during the test: {e}")

