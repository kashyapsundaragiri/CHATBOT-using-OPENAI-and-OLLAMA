import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.title("RAG Document Q&A")
st.write("Upload-free PDF Q&A over the papers in `research_papers/`")

groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

if not groq_api_key or not openai_api_key:
    st.warning("Set `GROQ_API_KEY` and `OPENAI_API_KEY` in your `.env` file, then restart the app.")
    st.stop()

llm = ChatGroq(model="openai/gpt-oss-20b", api_key=groq_api_key)

prompt_template = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question: {input}
"""
)

DATA_DIR = Path(__file__).parent / "research_papers"


def create_vector_embeddings():
    if "vector_store" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        st.session_state.loader = PyPDFDirectoryLoader(str(DATA_DIR))
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.documents[:50]
        )
        st.session_state.vector_store = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )


user_prompt = st.text_input("Enter your question:")

if st.button("Document Embedding"):
    with st.spinner("Creating embeddings..."):
        create_vector_embeddings()
    st.success("Document embedding complete.")

if user_prompt:
    if "vector_store" not in st.session_state:
        st.info("Click **Document Embedding** first to index the PDFs.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = st.session_state.vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("Thinking..."):
            start_time = time.process_time()
            response = retrieval_chain.invoke({"input": user_prompt})
            st.caption(f"Response time: {time.process_time() - start_time:.2f}s")
            st.write(response["answer"])

            with st.expander("Document Similarity Search"):
                for doc in response["context"]:
                    st.write(doc.page_content)
                    st.write("---")
