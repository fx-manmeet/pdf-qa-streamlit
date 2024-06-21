import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from groq import Groq
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
import os

with st.sidebar:
    st.title('PDF Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Gorq-mixtral-8x7b-32768](https://platform.openai.com/docs/models) LLM model
 
    ''')

    add_vertical_space(5)
    st.write('Ask anything!')

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )

llm = ChatGroq(temperature=0,
                          model_name="mixtral-8x7b-32768",
                          api_key=GROQ_API_KEY)

embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

custom_prompt_template = """Use the following pieces of information to answer the user's question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context: {context}
            Question: {question}

            Only return the helpful answer below and nothing else.
            Helpful answer:
            """

prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

def main():
    st.header("Chat with pdf")

    #upload the file
    pdf= st.file_uploader("Upload your pdf", type='pdf')
    # st.write(pdf)

    if pdf is not None:
        pdfreader = PdfReader(pdf)
        # st.write(pdfreader)

        text = ""
        for page in pdfreader.pages:
            text+=page.extract_text()

        # st.write(text)

        
        chunks = text_splitter.split_text(text=text)
        docs = [Document(page_content=chunk) for chunk in chunks]  #chromadb likes it in doc
        # st.write(chunks)
        
        storename=pdf.name[:-4]
        if os.path.exists(f"chroma_db_llamaparse{storename}"):
            vectorstore = Chroma(embedding_function=embed_model,
                         persist_directory=f"chroma_db_llamaparse{storename}",
                         collection_name=f"rag1{storename}")
            st.write('Embeddings Loaded from the Disk')
        else:
            vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embed_model,
            persist_directory=f"chroma_db_llamaparse{storename}",
            collection_name=f"rag1{storename}")


        query = st.text_input("Ask questions about your PDF file:")

        if query:
            
            
            retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

            
            qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={"prompt": prompt})
            
            response = qa.invoke({"query": query})
            st.write(response['result'])





if __name__=='__main__':
    main()