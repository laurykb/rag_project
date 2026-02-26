# retrieval/keyword_bm25.py
# Recherche multi-queries BM25 : helpers et API simple pour scorer et extraire les résultats pertinents
from indexing.keyword_index import bm25_search
from config import NUM_CHUNKS
# Extrait la liste ordonnée d'IDs depuis un résultat BM25
def _ranked_ids_from_result(res):
    return res["ids"][0] if res and res.get("ids") else []

# Transforme un résultat BM25 en lookup (recherche) : id -> {doc, meta, bm25}
def search_from_result(res):
    """
    La fonction search_from_result transforme le résultat brut d’une recherche BM25 (ou Chroma) en un dictionnaire pratique : 
    pour chaque identifiant de document trouvé, elle associe un petit dictionnaire contenant le texte du document (doc), ses métadonnées (meta) et son score (bm25 pour BM25 ou distance pour Chroma).
    """
    out = {}
    if not res or not res.get("ids"):
        return out
    ids = res["ids"][0] # IDs des résultats en fonction de la requête
    docs = res["documents"][0] # Textes des documents en fonction de la requête
    metas = res["metadatas"][0] # Métadonnées des documents en fonction de la requête
    scores = res.get("scores", [[]])[0] if res.get("scores") else None
    for i, id in enumerate(ids):
        out[id] = {"doc": docs[i], "meta": metas[i], "bm25": float(scores[i]) if scores is not None else None} # retourne en fonction de l'ID le texte, les métadonnées et le score BM25
    return out

# Recherche BM25 pour une seule requête : retourne (ids triés, lookup)
def run_bm25_for_query(bm25_tuple, query, topn=NUM_CHUNKS, source_filter: str = None): # bm25_tuple = (bm25_index, ids, texts, metadatas)
    bm25_index, ids, texts, metadatas = bm25_tuple
    # Filtrer par source si demandé
    if source_filter:
        filtered = [(i, t, m) for i, t, m in zip(ids, texts, metadatas) if m.get("source") == source_filter]
        if filtered:
            ids_f, texts_f, metas_f = zip(*filtered)
            res = bm25_search(bm25_index, list(ids_f), list(texts_f), list(metas_f), query, topn=topn)
        else:
            return [], {}
    else:
        res = bm25_search(bm25_index, ids, texts, metadatas, query, topn=topn)
    return _ranked_ids_from_result(res), search_from_result(res)

