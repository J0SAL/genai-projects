from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class VectorStore:
    def __init__(self, path):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", transport="grpc")
        self.vector_store = Chroma(
            persist_directory=path,
            embedding_function=self.embeddings
        )

    def add_documents(self, documents):
        for document in documents:
            document.metadata = self._sanitize_metadata(document.metadata)
        self.vector_store.add_documents(documents)
        
    def similarity_search(self, query, k=4, fetch_k=None, role=None, rbac_service=None):
        documents = self.vector_store.similarity_search(query, k=fetch_k or k)
        if role and rbac_service:
            documents = rbac_service.filter_documents(documents, role)
        return documents[:k]

    def _sanitize_metadata(self, metadata):
        sanitized = {}
        for key, value in (metadata or {}).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = ",".join(str(item) for item in value)
            else:
                sanitized[key] = str(value)
        return sanitized
