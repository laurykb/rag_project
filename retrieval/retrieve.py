# retrieve.py
# Pipeline hybride : semantic search, BM25, GraphRAG, fusion RRF, rerank Cross-Encoder, Parent-Child
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from retrieval.semantic_search import run_semantic_for_query
from retrieval.keyword_bm25 import run_bm25_for_query
from retrieval.rrf import fuse_with_rrf
from retrieval.cross_encoder import rerank_cross_encoder
from retrieval.graph_retrieve import graph_retrieve_to_lookup
from retrieval.parent_child import expand_to_parent
from utils.debug_utils import print_simple_results
from config import USE_CROSS_ENCODER, CROSS_ENCODER_LOCAL_PATH, CE_DEVICE, NUM_CHUNKS, RRF_K, WEIGHT_SEMANTIC, WEIGHT_BM25, PARENT_CHILD_ENABLED, CE_RELEVANCE_THRESHOLD

# Recherche hybride : semantic + BM25 + GraphRAG + fusion RRF + rerank Cross-Encoder

def hybrid_retrieve(collection, query, bm25_tuple, topk_chunks=NUM_CHUNKS, rrf_k=RRF_K, rerank_on=True, debug=True, weight_semantic=WEIGHT_SEMANTIC, weight_bm25=WEIGHT_BM25, entity_graph=None, source_filter: str = None, parent_child_on: bool = None):
    """
    Retourne (fused_chunks, max_ce_score).
    max_ce_score : score CE maximum observé sur tous les chunks après rerank.
                   None si le cross-encoder n'a pas tourné (USE_CROSS_ENCODER=False).
    """
    _t_start = time.perf_counter()

    # ── Étapes 1, 2, 3 lancées EN PARALLÈLE (elles sont totalement indépendantes) ──
    sem_ids, sem_lookup = [], {}
    bm_ids,  bm_lookup  = [], {}
    graph_ids, graph_lookup = [], {}

    def _run_semantic():
        return run_semantic_for_query(collection, query, topn=topk_chunks, source_filter=source_filter)

    def _run_bm25():
        if bm25_tuple:
            return run_bm25_for_query(bm25_tuple, query, topn=topk_chunks, source_filter=source_filter)
        return [], {}

    def _run_graph():
        if entity_graph is not None:
            return graph_retrieve_to_lookup(
                entity_graph, query, chunks_collection=None,
                max_hops=2, max_chunks=topk_chunks
            )
        return [], {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        fut_sem   = executor.submit(_run_semantic)
        fut_bm25  = executor.submit(_run_bm25)
        fut_graph = executor.submit(_run_graph)
        sem_ids,   sem_lookup   = fut_sem.result()
        bm_ids,    bm_lookup    = fut_bm25.result()
        graph_ids, graph_lookup = fut_graph.result()
    _t_retrieval = time.perf_counter()
    print(f"[⏱ retrieval] parallel search : {(_t_retrieval - _t_start)*1000:.0f}ms"
          f"  (sem={len(sem_ids)} bm25={len(bm_ids)} graph={len(graph_ids)})")

    # 1) Post-processing sémantique
    for i in sem_ids:
        if i in sem_lookup and sem_lookup[i].get("distance") is not None:
            sem_lookup[i]["sim_est"] = 1.0 - float(sem_lookup[i]["distance"])
    if debug:
        print_simple_results("Semantic results", [sem_lookup.get(i, {}) for i in sem_ids], max_items=topk_chunks)

    # 2) Debug BM25
    if debug and bm_ids:
        print_simple_results("BM25 results", [bm_lookup.get(i, {}) for i in bm_ids], max_items=topk_chunks)

    # 3) Debug GraphRAG
    if debug and entity_graph is not None:
        if graph_ids:
            exclusive = [gid for gid in graph_ids if gid not in sem_ids and gid not in bm_ids]
            print(f"[graph_retrieve] {len(graph_ids)} chunks total, "
                  f"{len(exclusive)} exclusifs (non trouvés par sémantique/BM25)")
            print_simple_results("GraphRAG results", [graph_lookup.get(i, {}) for i in graph_ids], max_items=topk_chunks)
        else:
            print("[graph_retrieve] Aucun chunk trouvé via le graphe")

    if not sem_ids and not bm_ids and not graph_ids:
        return [], None

    # 4) Fusion RRF — le graphe est fusionné comme source supplémentaire côté sémantique
    #    (les chunks graph sont des "bonus" qui enrichissent le pool sémantique)
    all_sem_ids = sem_ids + [gid for gid in graph_ids if gid not in sem_ids]
    all_sem_lookup = {**sem_lookup, **graph_lookup}

    fused = fuse_with_rrf(
        lists_a=[all_sem_ids], lookups_a=[all_sem_lookup],
        lists_b=[bm_ids] if bm_ids else None, lookups_b=[bm_lookup] if bm_lookup else None,
        rrf_k=rrf_k, topk_final=topk_chunks,
        weight_semantic=weight_semantic, weight_bm25=weight_bm25
    )
    if debug:
        print_simple_results("Fusion RRF", fused, max_items=topk_chunks)
    # 5) Rerank Cross-Encoder
    max_ce_score = None
    if rerank_on and USE_CROSS_ENCODER and fused:
        _t_before_ce = time.perf_counter()
        fused = rerank_cross_encoder(query, fused, model_path=CROSS_ENCODER_LOCAL_PATH, device=CE_DEVICE)
        _t_after_ce = time.perf_counter()
        print(f"[⏱ retrieval] cross-encoder rerank : {(_t_after_ce - _t_before_ce)*1000:.0f}ms")
        # Score CE max observé → utilisé pour la détection hors-scope
        if fused:
            max_ce_score = max(it.get("ce_score", 0.0) for it in fused)
            print(f"[scope] max CE score : {max_ce_score:.3f} (seuil={CE_RELEVANCE_THRESHOLD})")
        if debug:
            print_simple_results("Classement final", fused, max_items=topk_chunks)

    # 6) Parent-Child : remplace le contenu enfant par la section parente complète
    # parent_child_on=None → utilise la valeur de config ; True/False → override
    _pc_active = PARENT_CHILD_ENABLED if parent_child_on is None else parent_child_on
    if _pc_active and fused:
        fused = expand_to_parent(fused)
        if debug:
            expanded = sum(1 for it in fused if it.get("parent_expanded"))
            print(f"[parent_child] {expanded}/{len(fused)} chunks étendus au contexte parent")

    _t_end = time.perf_counter()
    print(f"[⏱ retrieval] TOTAL pipeline retrieval : {(_t_end - _t_start)*1000:.0f}ms → {len(fused)} chunks")
    return fused, max_ce_score
