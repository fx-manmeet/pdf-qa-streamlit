import os
from langchain_community.vectorstores import Chroma

def initialize_text_splitter():
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

def create_vector_store(docs, embed_model, pdf_name):
    storename = pdf_name[:-4]
    persist_directory = f"chroma_db_llamaparse{storename}"
    collection_name = f"rag1{storename}"
    
    if os.path.exists(persist_directory):
        vectorstore = Chroma(
            embedding_function=embed_model,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embed_model,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    return vectorstore
