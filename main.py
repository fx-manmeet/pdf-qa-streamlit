import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from sidebar import setup_sidebar
from pdf_utils import extract_text_from_pdf
from vectorstore_utils import initialize_text_splitter, create_vector_store
from config import load_api_key, initialize_llm, initialize_embed_model

text_splitter = initialize_text_splitter()
api_key = load_api_key()
llm = initialize_llm(api_key)
embed_model = initialize_embed_model()

#define structure of prompt
custom_prompt_template = """Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:
        """
prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

def main():
    setup_sidebar()

    st.header("Chat with PDF")

    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:

        text = extract_text_from_pdf(pdf)
        
        chunks = text_splitter.split_text(text=text)
        
        docs = [Document(page_content=chunk) for chunk in chunks]        #Chromadb likes input in Document schema

        vectorstore = create_vector_store(docs, embed_model, pdf.name)
        
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})      #retrive top k context
        
        query = st.text_input("Ask questions about your PDF file:")
        
        if query:
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            
            response = qa.invoke({"query": query})
            st.write(response['result'])


if __name__ == '__main__':
    main()
