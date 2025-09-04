# RAG with Gemini and Streamlit

This is a simple question-answering application built with Streamlit that uses a Retrieval-Augmented Generation (RAG) pipeline. It fetches content from web pages, embeds it using Google Gemini, stores it in a Chroma vector store, and answers user questions based on the retrieved context.

## Features

- **Web Content Ingestion**: Loads data from specified URLs using `langchain_community.document_loaders.WebBaseLoader`.
- **Text Splitting**: Splits documents into manageable chunks for embedding.
- **Vector Embeddings**: Uses Google's `gemini-embedding-001` model to create vector representations of the text.
- **Vector Storage**: Stores and retrieves vectors using `ChromaDB`.
- **Question Answering**: Uses Google's `gemini-1.5-flash` model to generate answers based on user queries and retrieved context.
- **Web Interface**: A simple chat interface built with Streamlit.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    streamlit
    langchain
    langchain-community
    langchain-google-genai
    langchain-chroma
    python-dotenv
    beautifulsoup4
    ```
    Then, install the packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root of your project and add your Google API key:
    ```
    GOOGLE_API_KEY="your_google_api_key_here"
    ```

## How to Run

Execute the following command in your terminal to start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your web browser. You can then ask questions in the chat input box.