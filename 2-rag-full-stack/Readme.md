# Full-Stack RAG Application with Flask and Gemini

This project is a full-stack web application that implements a Retrieval-Augmented Generation (RAG) pipeline. It allows users to upload documents (.pdf, .txt) and ask questions about their content through a chat interface. The backend is built with Flask, and it uses Google Gemini for language understanding and generation, Chroma for vector storage, and AWS S3 for file backup.

## Features

-   **File Upload**: Supports `.pdf` and `.txt` document uploads.
-   **Document Processing**: Splits documents into manageable chunks for efficient retrieval.
-   **Vector Embeddings**: Creates vector representations of text using Google's `gemini-embedding-001` model.
-   **Vector Storage**: Stores and retrieves document embeddings using a local Chroma vector store.
-   **Cloud Backup**: Uploads the original documents to an AWS S3 bucket.
-   **Conversational Q&A**: Generates answers using Google's `gemini-1.5-flash` model, considering the chat history for contextual follow-up questions.
-   **Web Interface**: A clean and simple user interface built with HTML, CSS, and vanilla JavaScript.

## Project Structure

```
2-rag-full-stack/
├── app/
│   ├── main.py             # Flask application entry point, handles routing
│   ├── config.py           # Configuration loader from environment variables
│   ├── models/
│   │   └── vector_store.py # Manages the Chroma vector store
│   ├── services/
│   │   ├── llm_service.py    # Handles interaction with the Gemini LLM
│   │   └── storage_service.py# Manages file uploads to AWS S3
│   └── templates/
│       └── index.html      # Frontend HTML and JavaScript
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (credentials, etc.)
└── vector_db/              # Directory for the persistent Chroma vector store
```

### output

<img width="1574" height="1386" alt="image" src="https://github.com/user-attachments/assets/b2d3c1ce-99c8-4576-82ec-02a8e3531e26" />

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd 2-rag-full-stack
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the `2-rag-full-stack` directory and add your credentials. You can use the existing `.env` file as a template.
    ```env
    # filepath: 2-rag-full-stack/.env
    GOOGLE_API_KEY="your_google_api_key"
    AWS_ACCESS_KEY="your_aws_access_key"
    AWS_SECRET_KEY="your_aws_secret_key"
    AWS_BUCKET_NAME="your_s3_bucket_name"
    ```

## How to Run

Execute the following command in your terminal to start the Flask application:

```bash
python app/main.py
```

The application will be available at `http://0.0.0.0:8080`. Open this URL in your web browser.

## How to Use

1.  **Upload a Document**: Click "Choose a file..." to select a `.pdf` or `.txt` file, then click "Upload". The application will process the file and store its contents.
2.  **Ask Questions**: Once the document is processed, the chat input will be enabled. Type your question and press "Send" to get an answer based on the document's content.