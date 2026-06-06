from tools import load_text_files, pdf_to_text, sentence_based_chunking

from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.document import Document

pdf_path = "pdf/"
text_path = "text/"

pdf_to_text(pdf_folder=pdf_path,output_folder=text_path)
text = load_text_files(text_path)
chunks = sentence_based_chunking(text)

documents = [Document(page_content=chunk["text"], metadata=chunk["metadata"]) for chunk in chunks]
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory="./chroma_db")
print("Chunks stored in ChromaDB.")