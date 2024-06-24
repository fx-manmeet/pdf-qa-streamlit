# PDF Chat App
## Introduction
PDF Chat App is a powerful tool that allows users to upload PDF files and ask questions about the content using an LLM-powered chatbot. The app leverages various advanced technologies including Streamlit, LangChain, and the Groq-mixtral-8x7b-32768 LLM model to provide accurate and helpful responses to user queries.

## Features
**Upload PDF Files:** Users can upload PDF files to the app.(`Streamlit`)

**Text Extraction:** The app extracts text from the uploaded PDF files.(`PyPDF2`)

**Text Chunking:** The extracted text is chunked into smaller pieces for efficient processing.(`LangChain`)

**Vector Store Creation:** Creates a vector store from the chunks of text for efficient retrieval.(`Chroma`)(`FastEmbedEmbeddings`)

**Persistent Storage:** If a vector store for the PDF already exists, it loads the vector store to save time.

**Interactive Q&A:** Users can ask questions about the PDF and receive answers based on the extracted and chunked text.

**LLM-Powered:** Utilizes the Groq-mixtral-8x7b-32768 LLM model for generating responses.(`Groq-mixtral-8x7b-32768 LLM Model`)

## Installation
**Clone the Repository:**

`git clone https://github.com/fx-manmeet/pdf-chat-app.git`

`cd pdf-chat-app`

**Create a Virtual Environment:**(for windows)

`python -m venv myvenv`

`myvenv/Scripts/activate`

**Install Dependencies:**

`pip install -r requirements.txt`

**Set Up Environment Variables:**

Create a `.env` file in the root directory of the project.

Add your Groq API key in the `.env` file: `GROQ_API_KEY=your_groq_api_key`

## Usage
**Run the App:**

`streamlit run main.py`

**Upload a PDF:**

Use brouse files or drag and drop to upload a PDF file.

**Ask Questions:**

Type your question in the text input box and press Enter.

The app will display the answer based on the content of the uploaded PDF.

## Acknowledgments

Streamlit

LangChain

Groq

PyPDF2

Chroma

Feel free to ask any questions or provide feedback to help improve this project!
