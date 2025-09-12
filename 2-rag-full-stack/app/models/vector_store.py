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
        self.vector_store.add_documents(documents)
        
    def similarity_search(self, query, k=4):
        return self.vector_store.similarity_search(query, k=k)