import fnmatch
import json
import os


class RBACService:
    def __init__(self, policy_path="data/access_policies.json"):
        self.policy_path = policy_path
        self.policies = self._load_policies()

    def _load_policies(self):
        if os.path.exists(self.policy_path):
            with open(self.policy_path, "r", encoding="utf-8") as policy_file:
                return json.load(policy_file)

        return {
            "users": {
                "avery_admin": {"name": "Avery Admin", "role": "admin"},
                "alex_analyst": {"name": "Alex Analyst", "role": "employee"},
            },
            "source_policies": [
                {
                    "pattern": "*",
                    "department": "general",
                    "sensitivity": "internal",
                    "allowed_roles": ["employee", "admin"],
                }
            ],
        }

    def get_users(self):
        return self.policies.get("users", {})

    def get_user(self, user_id):
        users = self.get_users()
        return users.get(user_id) or next(iter(users.values()), {"role": "employee"})

    def get_user_role(self, user_id):
        return self.get_user(user_id).get("role", "employee")

    def policy_for_source(self, source_name):
        for policy in self.policies.get("source_policies", []):
            if fnmatch.fnmatch(source_name.lower(), policy.get("pattern", "*").lower()):
                return policy

        return {
            "department": "general",
            "sensitivity": "internal",
            "allowed_roles": ["employee", "admin"],
        }

    def apply_policy_metadata(self, document, source_name):
        policy = self.policy_for_source(source_name)
        metadata = dict(document.metadata or {})
        allowed_roles = policy.get("allowed_roles", ["employee", "admin"])
        metadata.update(
            {
                "source": source_name,
                "department": policy.get("department", "general"),
                "sensitivity": policy.get("sensitivity", "internal"),
                "allowed_roles": ",".join(allowed_roles),
            }
        )
        document.metadata = metadata
        return document

    def can_access_document(self, document, role):
        if role == "admin":
            return True

        allowed_roles = document.metadata.get("allowed_roles", "")
        allowed = {item.strip() for item in allowed_roles.split(",") if item.strip()}
        return role in allowed

    def filter_documents(self, documents, role):
        return [doc for doc in documents if self.can_access_document(doc, role)]
