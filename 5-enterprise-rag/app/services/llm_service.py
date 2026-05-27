from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI


class LLMService:
    def __init__(self, vector_store, rbac_service):
        self.vector_store = vector_store
        self.rbac_service = rbac_service
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=700,
        )
        self.chat_history_by_user = {}
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a secure enterprise RAG assistant. Answer only from "
                    "the accessible context below. If the context is missing or "
                    "insufficient, say you do not have enough accessible information. "
                    "Do not reveal or infer restricted information. Include source "
                    "citations in the answer when relevant.\n\n"
                    "User role: {role}\n"
                    "Accessible context:\n{context}",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def get_response(self, query, user_id="alex_analyst", role="employee"):
        route = self._route_query(query)
        documents = self.vector_store.similarity_search(
            query,
            k=8,
            fetch_k=24,
            role=role,
            rbac_service=self.rbac_service,
        )
        documents = self._apply_route(documents, route)

        if not documents:
            return {
                "answer": (
                    "I do not have enough accessible information to answer that. "
                    "The relevant source may be missing or restricted for your role."
                ),
                "citations": [],
                "confidence": "low",
                "trace": {
                    "route": route,
                    "retrieved_documents": 0,
                    "role": role,
                },
            }

        context = self._format_context(documents)
        citations = self._citations(documents)
        confidence = self._confidence(documents)
        history = self.chat_history_by_user.setdefault(user_id, [])

        try:
            messages = self.qa_prompt.format_messages(
                role=role,
                context=context,
                chat_history=history,
                input=query,
            )
            response = self.llm.invoke(messages)
            answer = response.content
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            answer = self._fallback_answer(documents)

        history.extend([
            HumanMessage(content=query),
            AIMessage(content=answer),
        ])

        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "trace": {
                "route": route,
                "retrieved_documents": len(documents),
                "role": role,
                "sources": citations,
            },
        }

    def _route_query(self, query):
        query_lower = query.lower()
        routes = {
            "csv": ["revenue", "finance", "budget", "cost", "sales", "database", "csv"],
            "json": ["log", "alert", "security", "audit trail", "event", "json"],
            "sql": ["compliance", "audit", "control", "sql", "record"],
            "txt": ["policy", "procedure", "report", "operations", "employee"],
            "pdf": ["pdf", "document", "paper", "resume"],
        }
        matched = [
            source_type
            for source_type, keywords in routes.items()
            if any(keyword in query_lower for keyword in keywords)
        ]
        return matched or ["semantic"]

    def _apply_route(self, documents, route):
        if route == ["semantic"]:
            return documents

        routed_documents = [
            document
            for document in documents
            if document.metadata.get("source_type") in route
            or any(item in document.metadata.get("department", "") for item in route)
        ]
        return routed_documents or documents

    def _format_context(self, documents):
        context_blocks = []
        for index, document in enumerate(documents, start=1):
            citation = document.metadata.get("citation", document.metadata.get("source", "unknown"))
            sensitivity = document.metadata.get("sensitivity", "internal")
            context_blocks.append(
                f"[{index}] Source: {citation} | Sensitivity: {sensitivity}\n"
                f"{document.page_content}"
            )
        return "\n\n".join(context_blocks)

    def _citations(self, documents):
        citations = []
        for document in documents:
            citation = document.metadata.get("citation") or document.metadata.get("source")
            if citation and citation not in citations:
                citations.append(citation)
        return citations

    def _confidence(self, documents):
        if len(documents) >= 4:
            return "high"
        if len(documents) >= 2:
            return "medium"
        return "low"

    def _fallback_answer(self, documents):
        first = documents[0]
        citation = first.metadata.get("citation", first.metadata.get("source", "the accessible source"))
        snippet = first.page_content.replace("\n", " ")[:500]
        return f"Based on {citation}: {snippet}"
