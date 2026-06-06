import os
from langchain_community.document_loaders import PyPDFLoader
from nltk.tokenize import sent_tokenize 
from nltk import download
download('punkt_tab')


def pdf_to_text(pdf_folder, output_folder):
    """Convert all PDFs in a folder to text files."""
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(pdf_folder):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_folder, filename)
        txt_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")

        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            text = "\n".join([doc.page_content for doc in documents])

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Converted {filename} to {txt_path}")

        except Exception as e:
            print(f"Error converting {filename}: {e}")

def load_text_files(text_folder):
    """Load all text files from a folder into a list of strings."""
    texts = []
    for filename in os.listdir(text_folder):
        if not filename.endswith(".txt"):
            continue

        txt_path = os.path.join(text_folder, filename)
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
            texts.append((filename, text))  # Store filename + text

    return texts


def sentence_based_chunking(texts, sentences_per_chunk=4):
    all_chunks = []
    for filename, text in texts:
        sentences = sent_tokenize(text)
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = " ".join(sentences[i:i + sentences_per_chunk])
            chunks.append(chunk)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {"source": filename, "chunk_index": i}
            })
    return all_chunks
