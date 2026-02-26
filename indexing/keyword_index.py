# indexing/keyword_index.py
# Fonctions simples pour indexer des documents avec BM25 et effectuer une recherche par mots-clés
from rank_bm25 import BM25Okapi
import re
import pickle
from pymongo import MongoClient

# Tokenisation basique : minuscule, mots alphanumériques (accents inclus)
def _tokenize(text):
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)

# Construit un texte enrichi pour le BM25 : contenu + keywords + questions
def _build_enriched_text(doc):
    """
    Concatène le contenu du chunk avec ses mots-clés et questions (si disponibles).
    Les keywords et questions sont répétés pour leur donner plus de poids dans BM25.
    """
    text = getattr(doc, 'page_content', '') or ''
    meta = getattr(doc, 'metadata', {})
    
    parts = [text]
    
    # Ajouter les keywords (répétés 2x pour boost)
    keywords_str = meta.get("keywords_str", "")
    if keywords_str:
        parts.append(keywords_str)
        parts.append(keywords_str)  # Boost x2
    
    # Ajouter les questions (répétées 2x pour boost)
    questions_str = meta.get("questions_str", "")
    if questions_str:
        parts.append(questions_str)
        parts.append(questions_str)  # Boost x2
    
    return " ".join(parts)

# Construit un index BM25 à partir d'une liste de documents LangChain (liste de chunks)
def build_bm25_index(docs):
    texts = [getattr(d, 'page_content', '') or '' for d in docs] # Texte brut de chaque document
    ids = [getattr(d, 'metadata', {}).get('id', f"doc_{i:04d}") for i, d in enumerate(docs)] # IDs uniques
    metadatas = [getattr(d, 'metadata', {}) for d in docs] # Métadonnées associées
    
    # Corpus enrichi pour le BM25 (contenu + keywords + questions)
    enriched_texts = [_build_enriched_text(d) for d in docs]
    corpus_tokens = [_tokenize(t) for t in enriched_texts] # Tokenisation du corpus enrichi
    bm25 = BM25Okapi(corpus_tokens) # index BM25 construit
    return bm25, ids, texts, metadatas

# Recherche BM25 : retourne les topn résultats les plus pertinents
def bm25_search(bm25, ids, texts, metadatas, query, topn=10):
    q_tokens = _tokenize(query) #on tokenize la requête
    if not q_tokens:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "scores": [[]]} # si requête vide, pas de résultats
    scores = bm25.get_scores(q_tokens) # scores BM25 pour chaque document
    # On trie les documents par score décroissant et on garde les topn
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topn]
    return {
        "ids": [[ids[i] for i in order]],
        "documents": [[texts[i] for i in order]],
        "metadatas": [[metadatas[i] for i in order]],
        "scores": [[float(scores[i]) for i in order]],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  BM25 multi-document : stockage / chargement dans MongoDB
# ─────────────────────────────────────────────────────────────────────────────

def save_bm25_to_mongo(bm25_tuple, source_doc: str,
                       db_name="ragdb", collection_name="bm25_indexes"):
    """Sérialise l'index BM25 d'un document dans MongoDB (upsert par source_doc)."""
    client = MongoClient("mongodb://localhost:27017")
    col = client[db_name][collection_name]
    blob = pickle.dumps(bm25_tuple)
    col.update_one(
        {"source_doc": source_doc},
        {"$set": {"source_doc": source_doc, "index_blob": blob}},
        upsert=True
    )
    print(f"[BM25] Index sauvegardé dans MongoDB pour '{source_doc}'")


def load_bm25_from_mongo(source_doc: str = None,
                         db_name="ragdb", collection_name="bm25_indexes"):
    """
    Charge un ou plusieurs index BM25 depuis MongoDB.
    - source_doc=None  → fusionne tous les index disponibles en un seul tuple global.
    - source_doc=<str> → charge uniquement l'index du document demandé.
    Retourne un tuple (bm25, ids, texts, metadatas) ou None si absent.
    Résultat mis en cache en mémoire pour éviter la désérialisation MongoDB à chaque requête.
    """
    return _load_bm25_cached(source_doc, db_name, collection_name)


# ── Cache en mémoire pour les index BM25 ──────────────────────────────────────
_bm25_cache: dict = {}

def _load_bm25_cached(source_doc, db_name, collection_name):
    """Charge et met en cache le tuple BM25 (clé = source_doc ou '__all__')."""
    cache_key = source_doc or "__all__"
    if cache_key in _bm25_cache:
        return _bm25_cache[cache_key]

    result = _load_bm25_from_mongo_impl(source_doc, db_name, collection_name)
    if result is not None:
        _bm25_cache[cache_key] = result
    return result


def invalidate_bm25_cache(source_doc: str = None):
    """Invalide le cache BM25 (après ré-indexation). Sans argument → vide tout le cache."""
    global _bm25_cache
    if source_doc is None:
        _bm25_cache.clear()
    else:
        _bm25_cache.pop(source_doc, None)
        _bm25_cache.pop("__all__", None)  # Invalide aussi le cache global


def _load_bm25_from_mongo_impl(source_doc: str = None,
                                db_name="ragdb", collection_name="bm25_indexes"):
    client = MongoClient("mongodb://localhost:27017")
    col = client[db_name][collection_name]

    query = {"source_doc": source_doc} if source_doc else {}
    docs = list(col.find(query))

    if not docs:
        return None

    if len(docs) == 1:
        return pickle.loads(docs[0]["index_blob"])

    # Fusion : concaténer ids / texts / metadatas et reconstruire un BM25 global
    all_ids, all_texts, all_metas = [], [], []
    for d in docs:
        _, ids, texts, metas = pickle.loads(d["index_blob"])
        all_ids.extend(ids)
        all_texts.extend(texts)
        all_metas.extend(metas)

    # Reconstruire les tokens à partir des textes bruts (enrichis si disponibles)
    from dataclasses import dataclass

    @dataclass
    class _FakeDoc:
        page_content: str
        metadata: dict

    fake_docs = [_FakeDoc(t, m) for t, m in zip(all_texts, all_metas)]
    enriched = [_build_enriched_text(d) for d in fake_docs]
    corpus_tokens = [_tokenize(t) for t in enriched]
    bm25 = BM25Okapi(corpus_tokens)
    return bm25, all_ids, all_texts, all_metas


def list_bm25_sources(db_name="ragdb", collection_name="bm25_indexes"):
    """Retourne la liste des source_doc ayant un index BM25 en base."""
    client = MongoClient("mongodb://localhost:27017")
    return client[db_name][collection_name].distinct("source_doc")
