# Full-Stack RAG Application with Flask and Gemini

This project is a full-stack web application that implements a Retrieval-Augmented Generation (RAG) pipeline. It allows users to upload enterprise data sources and ask questions through a chat interface. The backend is built with Flask, and it uses Google Gemini for language understanding and generation, Chroma for vector storage, role-based access control (RBAC) for secure retrieval, and optional AWS S3 for original file backup.

## Features

-   **File Upload**: Supports `.pdf`, `.txt`, `.csv`, `.json`, and `.sql` uploads.
-   **Document Processing**: Splits documents into manageable chunks for efficient retrieval.
-   **Vector Embeddings**: Creates vector representations of text using Google's `gemini-embedding-001` model.
-   **Vector Storage**: Stores and retrieves document embeddings using a local Chroma vector store.
-   **RBAC Security**: Filters retrieved context by user role before the LLM receives it.
-   **Query Routing**: Routes finance, security, compliance, operations, document, and log-style questions toward relevant source types.
-   **Explainability**: Returns confidence, source citations, and a retrieval trace.
-   **Cloud Backup**: Optionally uploads the original documents to an AWS S3 bucket when AWS credentials are configured.
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
│   │   ├── enterprise_loader.py # Loads PDF, text, CSV, JSON, and SQL sources
│   │   ├── llm_service.py       # Handles secure grounded answers with citations
│   │   ├── rbac_service.py      # User-role and source access policy enforcement
│   │   └── storage_service.py   # Manages optional file uploads to AWS S3
│   └── templates/
│       └── index.html      # Frontend HTML and JavaScript
├── data/
│   ├── access_policies.json # User-role mappings and source access policies
│   └── enterprise/          # Synthetic CSV, JSON, SQL, and text records
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (credentials, etc.)
└── vector_db/              # Directory for the persistent Chroma vector store
```

## Enterprise Challenge Demo

1.  Start the app and open `http://0.0.0.0:8080`.
2.  Pick a user from **Query as**.
3.  Click **Load Demo Data** to index the synthetic enterprise silos in `data/enterprise`.
4.  Ask the same question as different users to verify RBAC behavior.

Example questions:

-   "What security incidents affected restricted systems?"
-   "Summarize Q2 finance risk."
-   "Which compliance controls are partial?"
-   "What operations bottleneck happened in May?"

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

1.  **Select a User**: Choose a user role to test RBAC.
2.  **Load Demo Data or Upload a Source**: Click "Load Demo Data" or upload a `.pdf`, `.txt`, `.csv`, `.json`, or `.sql` file.
3.  **Ask Questions**: Type a question and press "Send". The response includes citations, confidence, and retrieval trace metadata.
