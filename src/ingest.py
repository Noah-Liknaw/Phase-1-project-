# src/ingest.py

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

import os

def load_pdfs(pdf_paths):
    all_docs = []
    for path in pdf_paths:
        abs_path = os.path.abspath(path)
        print(f"🔎 Looking for PDF: {abs_path}")
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"❌ File not found: {abs_path}")
        loader = PyPDFLoader(abs_path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs

def load_pdfs(pdf_paths):
    """Load PDFs and return LangChain Documents"""
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs

def load_texts(text_paths):
    """Load plain text files (e.g., YouTube transcripts)"""
    all_docs = []
    for path in text_paths:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks for embeddings"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

if __name__ == "__main__":
    # --- Step 1: Set file paths ---
    pdf_folder = "../data/"
    txt_folder = "../data/"

    pdf_docs = load_pdfs([
        os.path.join(pdf_folder, "financialliteracy-ebook_mmi.pdf"),
        os.path.join(pdf_folder, "Money-Youth-2018-EN.pdf")
    ])

    txt_docs = load_text_files([
        os.path.join(txt_folder, "[English (auto-generated) (auto-generated)] Financial Literacy In 63 Minutes [DownSub.com].txt"),
        os.path.join(txt_folder, "[English (auto-generated) (auto-generated)] Learn Wealth-Building SECRETS in 30 Minutes _ Financial Literacy for Winners [DownSub.com].txt"),
        os.path.join(txt_folder, "[English (auto-generated) (auto-generated)] Master Financial Literacy in 54 Minutes_ Everything They Never Taught You About Money! [DownSub.com].txt")
    ])

    # --- Step 2: Combine and chunk ---
    all_docs = pdf_docs + txt_docs
    chunked_docs = chunk_documents(all_docs)

    # --- Step 3: Basic sanity checks ---
    print(f"Total documents loaded: {len(all_docs)}")
    print(f"Total chunks after splitting: {len(chunked_docs)}")

    # Print first 3 chunks to verify
    for i, chunk in enumerate(chunked_docs[:3]):
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content[:500])  # first 500 chars
        print("\n")
