import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

def setup_sidebar():
    with st.sidebar:
        st.title('PDF Chat App')
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [Gorq-mixtral-8x7b-32768 LLM model](https://console.groq.com/docs/libraries) 

        ''')
        add_vertical_space(5)
        st.write('Ask anything!')
