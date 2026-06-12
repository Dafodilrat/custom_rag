import os
from pypdf import PdfReader
from typing import List, Dict, Any


def convert_pdf_to_text(pdf_path: str) -> str:
    """Extracts all text from a PDF file as a single string."""
    reader = PdfReader(pdf_path)
    extracted_text = []
    
    # Loop through each page and extract text
    for page in reader.pages:
        text = page.extract_text()
        if text:  # Ensure the page isn't blank or an un-scanned image
            extracted_text.append(text)
            
    # Join all pages with a clean newline delimiter
    return "\n\n".join(extracted_text)

# Example Usage:
# text_content = pdf_to_text("path/to/document.pdf")


if __name__ == '__main__':
    # Example usage for testing
    TEST_FOLDER = "data_pdfs"
    if not os.path.exists(TEST_FOLDER):
        os.makedirs(TEST_FOLDER)
        print(f"Created test directory: {TEST_FOLDER}. Please add PDF files to test.")
    
    # This part requires actual PDFs to run successfully
    # chunks = chunker(TEST_FOLDER)
    # print(f"Example chunk structure: {chunks[0]}")
    print("Run the script with actual PDFs in 'data_pdfs' folder to test.")