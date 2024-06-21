import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

def load_api_key():
    load_dotenv()
    return os.getenv("GROQ_API_KEY")

def initialize_llm(api_key):
    return ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",
        api_key=api_key
    )

def initialize_embed_model():
    return FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
