# prerequisites:
# pip install langchain langchain-community langchain-openai chromadb pypdf streamlit

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

def process_pdf(file_path):
    # Load PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(pages)

    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )
    return vectorstore

def main():
    st.title("RAG System with PDF Upload")

    # Sidebar for API key and file upload
    with st.sidebar:
        st.header("Configuration")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file and openai_api_key:
        # Save uploaded file
        file_path = f"./temp/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process PDF
        vectorstore = process_pdf(file_path)

        # Initialize QA system
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # Question input
        question = st.text_input("Ask your question:")
        if question:
            result = qa.invoke({"query": question})
            st.write("Answer:", result["result"])

if __name__ == "__main__":
    main()