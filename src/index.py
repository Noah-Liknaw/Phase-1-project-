# src/index.py
import os
from ingest import load_texts
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# --- Helper function for loading PDFs with debug ---
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

# --- Define project directories ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# --- Load PDFs and text files ---
pdf_docs = load_pdfs([
    os.path.join(DATA_DIR, "financialliteracy-ebook_mmi.pdf"),
    os.path.join(DATA_DIR, "PersonalFinanceBook1.pdf")
])

txt_docs = load_texts([
    os.path.join(DATA_DIR, "[English (auto-generated) (auto-generated)] Financial Literacy In 63 Minutes [DownSub.com].txt"),
    os.path.join(DATA_DIR, "[English (auto-generated) (auto-generated)] Learn Wealth-Building SECRETS in 30 Minutes _ Financial Literacy for Winners [DownSub.com].txt"),
    os.path.join(DATA_DIR, "[English (auto-generated) (auto-generated)] Master Financial Literacy in 54 Minutes_ Everything They Never Taught You About Money! [DownSub.com].txt")
])

all_docs = pdf_docs + txt_docs
print(f"✅ Total documents loaded: {len(all_docs)}")

# --- Chunk documents ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_docs = splitter.split_documents(all_docs)
print(f"✅ Total chunks created: {len(chunked_docs)}")

# --- Create embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Build FAISS vectorstore ---
vectorstore = FAISS.from_documents(chunked_docs, embeddings)

# --- Save vectorstore locally ---
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
vectorstore.save_local(os.path.join(VECTORSTORE_DIR, "finance_index"))

print("✅ Vectorstore created and saved successfully!")
