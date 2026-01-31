import os
import pickle

import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.title("New research Tool")

st.sidebar.title("News article URL")
urls = []
file_path = "vectorstore.pkl"

llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 256}
    # model_kwargs={"temperature": 0}
)

for i in range(1):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

if process_url_clicked:
    # load URLs
    loader = UnstructuredURLLoader(urls=urls)
    # loader = TextLoader("test_context.txt")
    main_placeholder.text("Loading data from URLs...")
    data = loader.load()
    print(data)

    # Split Data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, separators=['\n\n', '\n', '.', ' '])
    main_placeholder.text("Splitting data into chunks...")
    docs = text_splitter.split_documents(data)

    # # Tokenizes and embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Creating vector store...")

    # Save vectorstore
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

query = main_placeholder.text_input("Question : ")

# python
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

            prompt = PromptTemplate(
                template="""
            Use the following context to answer the question.

            Rules:
            - Answer in ONE short phrase or sentence
            - Do NOT add explanations
            - If the answer is not in the context, say "Not found"

            Context:
            {context}

            Question:
            {question}

            Answer:
            """,
                input_variables=["context", "question"]
            )

            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )

            # Call the chain with the 'input' key to satisfy the combine_documents chain
            result = chain({"query": query})
            answer = result.get("result")
            source_docs = result.get("source_documents", [])

            st.header("Answer")
            st.write(answer)
