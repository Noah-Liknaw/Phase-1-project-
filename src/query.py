import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Dynamically find the path to the vectorstore folder relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_DIR = os.path.join(SCRIPT_DIR, "..", "vectorstore", "finance_index")

# Print paths for debugging
print("Script directory:", SCRIPT_DIR)
print("Looking for FAISS index at:", VECTORSTORE_DIR)

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Check if the FAISS index files exist
index_file = os.path.join(VECTORSTORE_DIR, "index.faiss")
if not os.path.exists(index_file):
    raise FileNotFoundError(
        f"FAISS index file not found at {index_file}. "
        "Make sure you've created the vectorstore before running this script."
    )

# Load the vectorstore
vectorstore = FAISS.load_local(
    VECTORSTORE_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

# Example query
query = "What is financial literacy?"
docs = vectorstore.similarity_search(query, k=3)

# Print results
for i, doc in enumerate(docs, 1):
    print(f"\n--- Document {i} ---\n")
    print(doc.page_content)
