# core/chat_sessions.py
"""
Gestion des sessions de conversation persistées dans MongoDB.
Collection : ragdb.chat_sessions

Chaque session contient :
  - _id         : ObjectId MongoDB
  - session_id  : str (raccourci lisible)
  - title       : str (auto-généré depuis la 1ère question, max 60 chars)
  - source_filter : str | None  (document filtré lors de la session)
  - created_at  : ISO timestamp
  - updated_at  : ISO timestamp
  - messages    : liste de {"role": "user"|"assistant", "content": str, "citations": list}
"""
from __future__ import annotations

import uuid
import time
from typing import Optional
from pymongo import MongoClient, DESCENDING


_MONGO_URI = "mongodb://localhost:27017"
_DB_NAME   = "ragdb"
_COL_NAME  = "chat_sessions"


def _col():
    client = MongoClient(_MONGO_URI)
    return client[_DB_NAME][_COL_NAME]


# ─────────────────────────────────────────────────────────────────────────────
#  CRUD
# ─────────────────────────────────────────────────────────────────────────────

def create_session(source_filter: str = None) -> str:
    """
    Crée une nouvelle session vide et retourne son session_id (str uuid court).
    """
    session_id = uuid.uuid4().hex[:12]
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    _col().insert_one({
        "session_id":    session_id,
        "title":         "Nouvelle conversation",
        "source_filter": source_filter,
        "created_at":    now,
        "updated_at":    now,
        "messages":      [],
    })
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    """Retourne la session complète (avec messages) ou None si introuvable."""
    doc = _col().find_one({"session_id": session_id})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


def list_sessions(limit: int = 50) -> list[dict]:
    """
    Retourne les sessions triées par date décroissante (sans les messages)
    pour affichage dans la barre latérale.
    """
    sessions = []
    for doc in _col().find({}, {"messages": 0}).sort("updated_at", DESCENDING).limit(limit):
        doc["_id"] = str(doc["_id"])
        sessions.append(doc)
    return sessions


def add_message(session_id: str, role: str, content: str, citations: list = None):
    """
    Ajoute un message à la session et met à jour le titre si c'est le 1er message user.
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    msg = {"role": role, "content": content, "citations": citations or []}

    # Auto-titre : 1ère question utilisateur (tronquée à 60 chars)
    if role == "user":
        session = _col().find_one({"session_id": session_id}, {"messages": 1, "title": 1})
        is_first_user = session and not any(
            m["role"] == "user" for m in session.get("messages", [])
        )
        if is_first_user:
            title = content[:60] + ("…" if len(content) > 60 else "")
            _col().update_one(
                {"session_id": session_id},
                {"$set": {"title": title, "updated_at": now}, "$push": {"messages": msg}},
            )
            return

    _col().update_one(
        {"session_id": session_id},
        {"$set": {"updated_at": now}, "$push": {"messages": msg}},
    )


def update_session_source(session_id: str, source_filter: Optional[str]):
    """Met à jour le filtre document d'une session."""
    _col().update_one(
        {"session_id": session_id},
        {"$set": {"source_filter": source_filter,
                  "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}},
    )


def delete_session(session_id: str):
    """Supprime définitivement une session."""
    _col().delete_one({"session_id": session_id})


def clear_session_messages(session_id: str):
    """Efface tous les messages d'une session (garde la session)."""
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    _col().update_one(
        {"session_id": session_id},
        {"$set": {"messages": [], "title": "Nouvelle conversation", "updated_at": now}},
    )


def get_messages(session_id: str) -> list[dict]:
    """Retourne uniquement les messages d'une session."""
    doc = _col().find_one({"session_id": session_id}, {"messages": 1})
    return doc.get("messages", []) if doc else []
