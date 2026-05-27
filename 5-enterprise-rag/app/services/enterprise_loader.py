import csv
import json
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EnterpriseDataLoader:
    def __init__(self, rbac_service):
        self.rbac_service = rbac_service
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

    def load_file(self, file_path, source_name=None):
        source_name = source_name or os.path.basename(file_path)
        extension = os.path.splitext(source_name)[1].lower()

        if extension == ".pdf":
            documents = PyPDFLoader(file_path).load()
        elif extension == ".txt":
            documents = TextLoader(file_path).load()
        elif extension == ".csv":
            documents = self._load_csv(file_path, source_name)
        elif extension == ".json":
            documents = self._load_json(file_path, source_name)
        elif extension == ".sql":
            documents = self._load_sql(file_path, source_name)
        else:
            raise ValueError("Unsupported file type")

        documents = [
            self.rbac_service.apply_policy_metadata(document, source_name)
            for document in documents
        ]
        for document in documents:
            document.metadata.setdefault("source_type", extension.lstrip("."))

        chunks = self.text_splitter.split_documents(documents)

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = index
            chunk.metadata["citation"] = self._build_citation(chunk.metadata)

        return chunks

    def load_directory(self, directory_path):
        documents = []
        supported_extensions = {".pdf", ".txt", ".csv", ".json", ".sql"}

        for root, _, files in os.walk(directory_path):
            for filename in sorted(files):
                if os.path.splitext(filename)[1].lower() in supported_extensions:
                    file_path = os.path.join(root, filename)
                    documents.extend(self.load_file(file_path, filename))

        return documents

    def _load_csv(self, file_path, source_name):
        documents = []
        with open(file_path, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row_number, row in enumerate(reader, start=1):
                content = "\n".join(f"{key}: {value}" for key, value in row.items())
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": source_name,
                            "source_type": "csv",
                            "record_id": row.get("id") or f"row-{row_number}",
                        },
                    )
                )
        return documents

    def _load_json(self, file_path, source_name):
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        records = data if isinstance(data, list) else data.get("records", [data])
        documents = []
        for index, record in enumerate(records, start=1):
            documents.append(
                Document(
                    page_content=json.dumps(record, indent=2, sort_keys=True),
                    metadata={
                        "source": source_name,
                        "source_type": "json",
                        "record_id": record.get("id") if isinstance(record, dict) else f"record-{index}",
                    },
                )
            )
        return documents

    def _load_sql(self, file_path, source_name):
        with open(file_path, "r", encoding="utf-8") as sql_file:
            statements = [
                statement.strip()
                for statement in sql_file.read().split(";")
                if statement.strip()
            ]

        return [
            Document(
                page_content=statement,
                metadata={
                    "source": source_name,
                    "source_type": "sql",
                    "record_id": f"statement-{index}",
                },
            )
            for index, statement in enumerate(statements, start=1)
        ]

    def _build_citation(self, metadata):
        source = metadata.get("source", "unknown source")
        record_id = metadata.get("record_id")
        page = metadata.get("page")

        if record_id:
            return f"{source}#{record_id}"
        if page is not None:
            return f"{source} page {page + 1}"
        return source
