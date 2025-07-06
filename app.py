# streamlit_app.py

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# === Streamlit UI ===
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 PDF-based RAG Chatbot with Ollama")

# === Upload PDF ===
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    with st.spinner("Reading and processing PDF..."):
        # === Load and split PDF ===
        loader = PyPDFLoader(uploaded_file.name)
        doc = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(doc)

        # === Embeddings ===
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # === Vector Store ===
        vectorstore = FAISS.from_documents(documents[:20], embedding=hf)
        retriever = vectorstore.as_retriever()

        # === LLM and Prompt ===
        llm = OllamaLLM(model="gemma3:1b")

        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        Think step by step before providing a detailed answer.
        I will tip you $1000 if the user finds the answer helpful,
        <context>
        {context}
        </context>
        Question: {input}""")

        document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

        # === Chat Interface ===
        st.success("PDF processed. You can now ask questions!")
        query = st.text_input("Ask a question from the PDF:")
        if query:
            docs = retriever.invoke(query)
            response = document_chain.invoke({"input": query, "context": docs})

            st.subheader("Answer:")
            st.write(response)

            with st.expander("Retrieved Documents"):
                for i, d in enumerate(docs):
                    st.markdown(f"**Doc {i+1}:**\n{d.page_content}")
