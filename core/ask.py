# ask.py
"""
Pose une question a la base RAG :
- Rewrite + Expansions (LLM local via Ollama)
- Retrieval hybride : Semantic (Chroma) + BM25 + GraphRAG
- Fusion RRF + Reranking Cross-Encoder
- Reponse finale (LLM) avec contexte construit

Pre-requis :
- Une collection Chroma deja indexee (ingest faite avant).
- Un cache BM25 (bm25_index.pkl) cree lors de l'ingestion.
"""

import os
import pickle
import chromadb
from typing import Optional, Tuple, Generator
from config import CHROMA_PATH, COLLECTION_NAME
from langchain_community.llms import Ollama
from config import (
    COLLECTION_NAME, NUM_CHUNKS, RRF_K,
    REWRITER_MODEL, WEIGHT_SEMANTIC, WEIGHT_BM25,
    SELF_RAG_ENABLED, NUM_CHUNKS_PARENT_CHILD,
    CE_RELEVANCE_THRESHOLD, OUT_OF_SCOPE_MESSAGE,
    USE_CROSS_ENCODER,
)
from retrieval.retrieve import hybrid_retrieve
from core.llm_answer import answer, answer_stream, build_context, build_citation_map
from nlp.vocab_builder import load_vocab
from nlp.query_rewriter import guarded_rewrite
from nlp.graph_builder import load_graph_from_mongo
from indexing.keyword_index import load_bm25_from_mongo

# ── Singletons / caches process-level ─────────────────────────────────────────
_chroma_client = None
_chroma_collection = None
_vocab_cache = None  # (vocab, acronyms)

def _get_chroma_collection():
    """Retourne la collection Chroma (singleton — connexion ouverte une seule fois)."""
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _chroma_collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return _chroma_collection

def _get_vocab():
    """Charge le vocabulaire une seule fois depuis le disque (cache en mémoire)."""
    global _vocab_cache
    if _vocab_cache is None:
        _vocab_cache = load_vocab("/home/marsattacks/Documents/RAG_Laury/data/vocab_save/vocab.json")
    return _vocab_cache

# ---------- Cache graphe (par source_doc) ----------
_entity_graph_cache: dict = {}  # source_doc -> graphe (ou False si indisponible)

def _load_entity_graph(source_doc: str = None):
    """
    Charge le graphe d'entités depuis MongoDB.
    - Si source_doc est fourni : charge le graphe de ce document (avec cache).
    - Si source_doc est None : charge le premier graphe disponible (fallback).
    Retourne None si aucun graphe n'est disponible.
    """
    global _entity_graph_cache
    cache_key = source_doc or "__first__"

    if cache_key not in _entity_graph_cache:
        try:
            from pymongo import MongoClient
            client = MongoClient("mongodb://localhost:27017")
            db = client["ragdb"]
            col = db["entity_graph"]

            if source_doc:
                # Chercher d'abord le graphe exact, puis par correspondance partielle
                target = source_doc
                available = col.distinct("source_doc")
                if source_doc not in available:
                    # Correspondance partielle : trouver le graphe dont le nom est le plus proche
                    matches = [d for d in available if source_doc.replace(".pdf", "").replace(".md", "") in d]
                    target = matches[0] if matches else (available[0] if available else None)
                    if target and target != source_doc:
                        print(f"[ask] Graphe exact introuvable pour '{source_doc}' → utilisation de '{target}'")
            else:
                available = col.distinct("source_doc")
                target = available[0] if available else None

            if target:
                graph = load_graph_from_mongo(source_doc=target)
                _entity_graph_cache[cache_key] = graph if graph else False
                if not graph:
                    print(f"[ask] Graphe vide pour '{target}' -> GraphRAG désactivé")
            else:
                _entity_graph_cache[cache_key] = False
                print("[ask] Aucun graphe en base -> GraphRAG désactivé")

        except Exception as e:
            print(f"[ask] Erreur chargement graphe : {e}")
            _entity_graph_cache[cache_key] = False

    result = _entity_graph_cache.get(cache_key, False)
    return result if result else None

# ---------- charger un index BM25 pre-calcule ----------
def load_bm25_cache(path: str = "data/bm25_index.pkl") -> Optional[Tuple]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            bm25_tuple = pickle.load(f)
        if isinstance(bm25_tuple, tuple) and len(bm25_tuple) == 4:
            return bm25_tuple
    except Exception as e:
        print("[ask] Impossible de charger le cache BM25 :", e)
    return None

def _condense_question(user_q: str, history: list[dict], llm_rewriter) -> str:
    """
    Si l'historique est non vide, reformule la question de suivi en une question
    autonome (standalone question) en tenant compte du contexte précédent.
    Ex: "Et pour la crypto ?" + historique → "Quelles sont les exigences crypto de FCS_CKM ?"
    """
    if not history or len(history) < 2:
        return user_q

    # Reconstruit les 2 derniers tours
    recent = history[-4:]
    turns = []
    for msg in recent:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        turns.append(f"{role}: {msg['content'][:400]}")
    history_str = "\n".join(turns)

    prompt = f"""Tu es un assistant qui reformule des questions de suivi en questions autonomes.

Historique de conversation :
{history_str}

Question de suivi : {user_q}

Reformule la question de suivi en une question autonome complète et précise,
qui peut être comprise sans l'historique. Réponds UNIQUEMENT avec la question reformulée,
sans explication, sans guillemets.

Question autonome :"""

    try:
        result = llm_rewriter.invoke(prompt).strip()
        # Garde la reformulation seulement si elle est pertinente
        if result and len(result) > 5 and result != user_q:
            print(f"[Condense] '{user_q}' → '{result}'")
            return result
    except Exception as e:
        print(f"[Condense] Erreur reformulation : {e}")
    return user_q


def process_query(user_q: str, selected_chunks=None, system_prompt=None, source_filter: str = None,
                  conversation_history: list = None):
    if not user_q:
        print("Aucune question fournie.")
        return None, None, None  # (response, chunks, citations)

    # Si des chunks sont fournis, ne faire que la generation de la reponse
    if selected_chunks is not None:
        rep, citations = answer(user_q, selected_chunks, system_prompt=system_prompt,
                                conversation_history=conversation_history)
        return rep, selected_chunks, citations

    # Chargement du vocabulaire
    vocab, acronyms = _get_vocab()

    # LLM local pour la reecriture
    llm_rewriter = Ollama(model=REWRITER_MODEL, temperature=0.2)

    # Reformulation standalone si question de suivi (mémoire conversationnelle)
    user_q_condensed = _condense_question(user_q, conversation_history or [], llm_rewriter)

    # Reecriture de la query en prenant en compte le vocab
    q_main = guarded_rewrite(user_q_condensed, llm_rewriter, vocab, acronyms)

    # On passe la query originale ET la query reecrite au retriever (concatenees)
    query = user_q_condensed.strip() + "\n" + q_main.strip()
    print("\n[Rewrite] :", q_main)

    # Ouvrir la collection Chroma deja indexee (singleton)
    collection = _get_chroma_collection()

    # Charger BM25 depuis MongoDB (multi-document) avec fallback pkl
    bm25_tuple = load_bm25_from_mongo(source_doc=source_filter)
    if bm25_tuple is None:
        # Fallback : pkl global
        import os as _os
        _bm25_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "bm25_index.pkl")
        bm25_tuple = load_bm25_cache(_bm25_path)
    if bm25_tuple is None:
        print("[Info] Aucun cache BM25 trouve -> retrieval sans keyword search.")

    # Charger le graphe d'entites (GraphRAG) — filtré par document si source_filter fourni
    entity_graph = _load_entity_graph(source_doc=source_filter)

    # Retrieve hybride (semantic + bm25 + GraphRAG) + RRF
    final_chunks, max_ce_score = hybrid_retrieve(
        collection=collection,
        query=query,
        bm25_tuple=bm25_tuple,
        topk_chunks=NUM_CHUNKS,
        rrf_k=RRF_K,
        rerank_on=True,
        debug=True,
        weight_semantic=WEIGHT_SEMANTIC,
        weight_bm25=WEIGHT_BM25,
        entity_graph=entity_graph,
        source_filter=source_filter
    )

    # ── Détection hors-scope ──────────────────────────────────────────────────
    if USE_CROSS_ENCODER and max_ce_score is not None and max_ce_score < CE_RELEVANCE_THRESHOLD:
        print(f"[scope] Hors-scope détecté (max CE={max_ce_score:.3f} < {CE_RELEVANCE_THRESHOLD})")
        return OUT_OF_SCOPE_MESSAGE, [], []

    if not final_chunks:
        print("\n[Resultat] Aucun chunk pertinent trouve.")
        return "Aucun chunk pertinent trouve.", [], []

    # Reponse finale (LLM de generation)
    rep, citations = answer(q_main, final_chunks, system_prompt=system_prompt,
                            conversation_history=conversation_history)
    print("\n--- Reponse ---\n")
    return rep, final_chunks, citations  # (reponse, chunks, citations)


def _prepare_retrieval(user_q: str, source_filter: str = None,
                       conversation_history: list = None,
                       parent_child_on: bool = None,
                       rewrite_enabled: bool = True):
    """
    Étapes communes de retrieval (rewrite + hybrid_retrieve).
    Gère la condensation standalone pour les questions de suivi.
    Retourne (q_main, final_chunks).
    """
    vocab, acronyms = _get_vocab()

    if rewrite_enabled:
        llm_rewriter = Ollama(model=REWRITER_MODEL, temperature=0.2, keep_alive=-1)
        # Reformulation standalone si question de suivi
        user_q_condensed = _condense_question(user_q, conversation_history or [], llm_rewriter)
        q_main = guarded_rewrite(user_q_condensed, llm_rewriter, vocab, acronyms)
        query = user_q_condensed.strip() + "\n" + q_main.strip()
        print("\n[Rewrite] :", q_main)
    else:
        # Mode rapide : pas de LLM rewriter, juste nettoyage léger
        from nlp.query_rewriter import trim_only_rewrite
        q_main = trim_only_rewrite(user_q)
        query = q_main
        print("\n[Rewrite] skipped (rewrite_enabled=False) :", q_main)

    collection = _get_chroma_collection()

    bm25_tuple = load_bm25_from_mongo(source_doc=source_filter)
    if bm25_tuple is None:
        import os as _os
        _bm25_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "bm25_index.pkl")
        bm25_tuple = load_bm25_cache(_bm25_path)

    # Charger le graphe d'entités filtré par document source
    entity_graph = _load_entity_graph(source_doc=source_filter)

    # Quand Parent-Child est actif, on réduit le topk pour éviter un contexte trop long
    # (chaque chunk enfant → ~4000 chars de contexte parent → 8 chunks → ~32K total)
    _pc_active = parent_child_on if parent_child_on is not None else True
    _topk = NUM_CHUNKS_PARENT_CHILD if _pc_active else NUM_CHUNKS

    final_chunks, max_ce_score = hybrid_retrieve(
        collection=collection,
        query=query,
        bm25_tuple=bm25_tuple,
        topk_chunks=_topk,
        rrf_k=RRF_K,
        rerank_on=True,
        debug=True,
        weight_semantic=WEIGHT_SEMANTIC,
        weight_bm25=WEIGHT_BM25,
        entity_graph=entity_graph,
        source_filter=source_filter,
        parent_child_on=parent_child_on,
    )

    # ── Détection hors-scope ──────────────────────────────────────────────────
    # Si le cross-encoder a tourné et que son meilleur score est sous le seuil,
    # aucun chunk n'est pertinent → on retourne un signal out_of_scope.
    if USE_CROSS_ENCODER and max_ce_score is not None and max_ce_score < CE_RELEVANCE_THRESHOLD:
        print(f"[scope] ⚠ Hors-scope détecté (max CE={max_ce_score:.3f} < {CE_RELEVANCE_THRESHOLD})")
        return q_main, [], max_ce_score  # final_chunks vide + score pour l'appelant

    return q_main, final_chunks


def process_query_stream(user_q: str, system_prompt=None, source_filter: str = None,
                         conversation_history: list = None,
                         parent_child_on: bool = None,
                         rewrite_enabled: bool = True):
    """
    Version streaming de process_query() avec mémoire conversationnelle.
    Si SELF_RAG_ENABLED, évalue et retente en non-streaming avant de streamer la meilleure réponse.
    Retourne (token_generator, final_chunks, citations).
    Si aucun chunk n'est trouvé, retourne (None, [], []).
    """
    if not user_q:
        return None, [], []

    # ── Self-RAG : évaluation + retry (non-streaming) ────────────────────────
    if SELF_RAG_ENABLED:
        from core.self_rag import self_rag_query
        best_answer, best_chunks, best_citations, self_rag_metrics = self_rag_query(
            user_q,
            system_prompt=system_prompt,
            source_filter=source_filter,
            conversation_history=conversation_history,
        )
        if not best_chunks:
            return None, [], []

        # On "streame" la réponse déjà générée token par token (simulé)
        def _replay_gen():
            for token in best_answer:
                yield token

        print(
            f"[Self-RAG] Score final : {self_rag_metrics.get('self_rag_score', 0):.2f} "
            f"en {self_rag_metrics.get('self_rag_attempts', 1)} tentative(s)"
        )
        return _replay_gen(), best_chunks, best_citations

    # ── Pipeline classique ────────────────────────────────────────────────────
    retrieval_result = _prepare_retrieval(
        user_q, source_filter=source_filter,
        conversation_history=conversation_history,
        parent_child_on=parent_child_on,
        rewrite_enabled=rewrite_enabled,
    )

    # _prepare_retrieval retourne (q_main, chunks) ou (q_main, [], max_ce_score) si hors-scope
    if len(retrieval_result) == 3:
        q_main, final_chunks, max_ce_score = retrieval_result
        # Hors-scope : on streame le message d'information
        def _scope_gen():
            yield OUT_OF_SCOPE_MESSAGE
        return _scope_gen(), [], []

    q_main, final_chunks = retrieval_result

    if not final_chunks:
        print("\n[Resultat] Aucun chunk pertinent trouve.")
        return None, [], []

    token_gen, citations = answer_stream(
        q_main, final_chunks, system_prompt=system_prompt,
        conversation_history=conversation_history
    )
    return token_gen, final_chunks, citations

if __name__ == "__main__":
    user_q = input("Question : ").strip()
    rep, chunks, citations = process_query(user_q)
    if rep:
        print(rep)
        if citations:
            print("\n--- Sources ---")
            for c in citations:
                page_str = f", page {c['page']}" if c['page'] else ""
                print(f"  [{c['idx']}] {c['source']}{page_str}")
