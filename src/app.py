import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Paths
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore", "finance_index")

# Load FAISS
@st.cache_resource
def load_vectorstore():
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)

# Load local Hugging Face model
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200,       
        min_length=50,        
        do_sample=True,       
        top_p=0.95,           
        temperature=0.7       
    )
    return HuggingFacePipeline(pipeline=pipe)

# Streamlit UI
st.set_page_config(page_title="Finance Document Search", page_icon="💰")

st.title("💰 Your Financial Assistant")
st.write("An assistant to help you understand concepts from finance transcripts and notes.")
st.write("⚠️ Not financial advice.😅")

# Layout: Two columns (image + input/output)
col1, col2 = st.columns([1, 2])  # left for image, right for Q&A

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", 
             caption="Finance Assistant", use_container_width=True)

with col2:
    query = st.text_input("Enter your question:")

    if query:
        vectorstore = load_vectorstore()

        # Get top 3 relevant documents
        docs = vectorstore.similarity_search(query, k=3)
        if not docs:
            st.subheader("Answer:")
            st.write("Did not find anything relevant.")
        else:
            # Take only first 4 sentences from each doc for context
            context_sentences = []
            for d in docs:
                sentences = d.page_content.split(". ")
                context_sentences.extend(sentences[:4])
            short_context = ". ".join(context_sentences)

            # Prompt template
            prompt_template = PromptTemplate(
                template=(
                    "You are a helpful financial assistant. Answer the question concisely in 2-3 sentences "
                    "using the context below. Ignore any irrelevant parts.\n\n"
                    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                ),
                input_variables=["context", "question"],
            )

            # Call the LLM
            llm = load_llm()
            chain = LLMChain(llm=llm, prompt=prompt_template)
            raw_answer = chain.run(question=query, context=short_context)

            # Return first 3–5 sentences
            sentences = raw_answer.split(". ")
            answer = ". ".join(sentences[:5]).strip()
            if not answer:
                answer = "Did not find anything relevant."

            st.subheader("Answer:")
            st.write(answer)
