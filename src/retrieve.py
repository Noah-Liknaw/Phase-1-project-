# src/retrieve.py

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def build_faiss_index():
    # --- Define where data is stored ---
    DATA_DIR = Path(__file__).parent.parent / "data"
    pdf_dir = DATA_DIR
    text_dir = DATA_DIR

    docs = []

    # --- Load PDFs ---
    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"📥 Loading PDF: {pdf_file}")
        loader = PyPDFLoader(str(pdf_file))
        docs.extend(loader.load())

    # --- Load text files ---
    for txt_file in text_dir.glob("*.txt"):
        print(f"📥 Loading TXT: {txt_file}")
        loader = TextLoader(str(txt_file), encoding="utf-8")
        docs.extend(loader.load())

    if not docs:
        print("⚠️ No documents found. Add PDFs or TXT files to the data/ folder.")
        return

    # --- Split into chunks ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"📄 Split into {len(split_docs)} chunks.")

    # --- Create embeddings ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # --- Build FAISS vectorstore ---
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    VECTORSTORE_PATH = Path(__file__).parent.parent / "vectorstore" / "finance_index"
    VECTORSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(VECTORSTORE_PATH))
    print(f"✅ Vectorstore saved at {VECTORSTORE_PATH}")


if __name__ == "__main__":
    build_faiss_index()
