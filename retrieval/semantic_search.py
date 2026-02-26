# retrieval/semantic_search.py
# Recherche sémantique multi-queries sur une collection ChromaDB avec embeddings
from nlp.ollama_embedding import OllamaEmbedding
from config import NUM_CHUNKS

# ── Singleton embedding (instancié une seule fois, réutilisé pour toutes les requêtes) ──
_embedding_singleton = None  # type: OllamaEmbedding | None

def _get_embedding_model():
    # type: () -> OllamaEmbedding
    global _embedding_singleton
    if _embedding_singleton is None:
        _embedding_singleton = OllamaEmbedding()
    return _embedding_singleton

# Extrait la liste ordonnée d'IDs depuis un résultat Chroma 
def _ranked_ids_from_result(res):
    return res["ids"][0] if res and res.get("ids") else []

# Transforme un résultat Chroma en lookup : id -> {doc, meta, distance}
def search_from_result(res):
    out = {}
    if not res or not res.get("ids"):
        return out
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[]])[0] if res.get("distances") else None
    for i, id in enumerate(ids):
        out[id] = {"doc": docs[i], "meta": metas[i], "distance": float(dists[i]) if dists is not None else None}
    return out

# Recherche sémantique Chroma pour une seule requête : retourne (ids triés, lookup)
def run_semantic_for_query(collection, query, topn=NUM_CHUNKS, embedding_model=None, source_filter: str = None):
    emb = embedding_model or _get_embedding_model()
    q_vec = emb.embed_query(query)
    kwargs = dict(
        query_embeddings=[q_vec],
        n_results=topn,
        include=["documents", "metadatas", "distances"]
    )
    if source_filter:
        kwargs["where"] = {"source": {"$eq": source_filter}}
    res = collection.query(**kwargs)
    return _ranked_ids_from_result(res), search_from_result(res)


