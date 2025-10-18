import os
import pickle

import langchain
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.title("New research Tool")

st.sidebar.title("News article URL")
urls = []
file_path = "vectorstore.pkl"

# llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

# llm = HuggingFacePipeline.from_model_id(model_id="distilgpt2", task="text-generation")

# llm = HuggingFacePipeline.from_model_id(
#     model_id="distilgpt2",
#     task="text-generation",
#     model_kwargs={"temperature": 1, "max_length": 1024},
#     pipeline_kwargs={"max_new_tokens": 500},
#     device=-1  # CPU (use 0 for GPU if available)
# )

llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 10},
)

for i in range(1):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

if process_url_clicked:
    # load URLs
    # loader = UnstructuredURLLoader(urls=urls)
    loader = TextLoader("test_context.txt")
    main_placeholder.text("Loading data from URLs...")
    data = loader.load()
    print(data)

    # Split Data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, separators=['\n\n', '\n', '.', ' '])
    main_placeholder.text("Splitting data into chunks...")
    docs = text_splitter.split_documents(data)

    # # Tokenizes and embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": False}
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Creating vector store...")

    # Save vectorstore
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

query = main_placeholder.text_input("Question : ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever()

            # Define prompt for the LLM
            prompt = ChatPromptTemplate.from_template(
                "Answer the question based on the context: {context}\nQuestion: {input}\nAnswer:"
            )
            # Create document chain and retrieval chain
            doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
            chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chain)
            langchain.debug = True
            # Fetch and display retrieved documents in a table
            docs = retriever.invoke(query)
            st.header("Documents Retrieved 1")
            doc_data = [{"Doc ID": f"Doc {i + 1}", "Content": doc.page_content} for i, doc in enumerate(docs)]
            df = pd.DataFrame(doc_data)
            st.dataframe(df, column_config={
                "Doc ID": st.column_config.TextColumn(width="small"),
                "Content": st.column_config.TextColumn(width="large")
            }, use_container_width=True)

            # Run chain and display answer
            result = chain.invoke({"input": query})

            st.header("Answer")
            st.subheader(result['context'][0].page_content)

            # st.header("Extra")
            # for index in range(len(result['context'])):
            #     st.subheader(result['context'][index].page_content)
