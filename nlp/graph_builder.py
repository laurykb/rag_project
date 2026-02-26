# nlp/graph_builder.py
"""
Étape 8 — GraphRAG : construction du graphe d'entités.

Construit un graphe orienté (NetworkX) à partir des entités extraites (Étape 7).
Deux types de relations :
  1) Co-occurrence : deux entités apparaissent dans le même chunk → lien
  2) LLM : le LLM extrait des relations typées (certifie, conforme_à, évalue, etc.)

Le graphe est persisté dans MongoDB (collection "entity_graph") et rechargé
au démarrage pour le retrieval.

Nœuds = entités (avec type PER/ORG/LOC/NORM/ACRO/MISC)
Arêtes = relations (avec type, source chunk_id, poids = nb co-occurrences)
"""

import json
import hashlib
import requests
import networkx as nx
from pathlib import Path
from typing import Optional
from pymongo import MongoClient
from config import REWRITER_MODEL

GRAPH_CACHE_DIR = Path("data/graph_cache")
DEFAULT_GRAPH_MODEL = REWRITER_MODEL


# ──────────────── Appel LLM pour extraction de relations ────────────────

RELATION_PROMPT = """Tu es un expert en extraction de relations entre entités dans des documents techniques.

Voici un texte et la liste des entités déjà identifiées.

Extrait les relations entre ces entités. Chaque relation doit avoir :
- source : l'entité source (exactement comme dans la liste)
- target : l'entité cible (exactement comme dans la liste)
- relation : le type de relation (un verbe ou expression courte, en français)

Types de relations courants : certifie, évalue, conforme_à, référence, développe,
supervise, délivre, applique, utilise, contient, appartient_à, remplace.

Réponds UNIQUEMENT en JSON, un tableau de relations :
[{{"source": "...", "target": "...", "relation": "..."}}, ...]

Si aucune relation n'est identifiable, réponds : []

Entités : {entities}

Texte :
{text}

Relations JSON :"""


def _call_ollama(prompt: str, model: str = None,
                 base_url: str = "http://localhost:11434") -> str:
    """Appel Ollama pour extraction de relations."""
    model = model or DEFAULT_GRAPH_MODEL
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 500,
        }
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"[graph_builder] Erreur Ollama : {e}")
        return ""


# ──────────────── Cache pour les relations LLM ────────────────

def _relation_cache_key(text: str, entities_str: str) -> str:
    h = hashlib.sha256(f"rel:{entities_str}:{text[:2000]}".encode()).hexdigest()[:16]
    return h


def _get_relation_cache(text: str, entities_str: str) -> Optional[list]:
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _relation_cache_key(text, entities_str)
    cache_file = GRAPH_CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return data.get("relations")
        except Exception:
            pass
    return None


def _set_relation_cache(text: str, entities_str: str, relations: list):
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _relation_cache_key(text, entities_str)
    cache_file = GRAPH_CACHE_DIR / f"{key}.json"
    try:
        cache_file.write_text(
            json.dumps({"relations": relations}, ensure_ascii=False),
            encoding="utf-8"
        )
    except Exception:
        pass


# ──────────────── Extraction de relations d'un chunk ────────────────

def extract_relations_from_chunk(text: str, entities_flat: list[str],
                                  chunk_id: str = "",
                                  model: str = None) -> list[dict]:
    """
    Extrait les relations entre entités d'un chunk via LLM.

    Retourne une liste de dicts :
      [{"source": "ANSSI", "target": "Mistral AI", "relation": "certifie",
        "chunk_id": "xxx"}, ...]
    """
    if not entities_flat or len(entities_flat) < 2:
        return []

    # Avec une seule entité, pas de relation possible
    entities_str = ", ".join(entities_flat[:15])  # Limiter

    # Cache
    cached = _get_relation_cache(text, entities_str)
    if cached is not None:
        # Ajouter chunk_id aux relations cachées
        for r in cached:
            r["chunk_id"] = chunk_id
        return cached

    prompt = RELATION_PROMPT.format(
        entities=entities_str,
        text=text[:3000]
    )
    raw = _call_ollama(prompt, model=model)

    relations = []
    if raw:
        # Parser le JSON — le LLM peut ajouter du texte autour
        try:
            # Chercher le tableau JSON dans la réponse
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                if isinstance(parsed, list):
                    for r in parsed:
                        if isinstance(r, dict) and "source" in r and "target" in r and "relation" in r:
                            # Valider que source et target sont dans les entités connues
                            src = r["source"].strip()
                            tgt = r["target"].strip()
                            rel = r["relation"].strip().lower().replace(" ", "_")
                            if src and tgt and rel and src != tgt:
                                relations.append({
                                    "source": src,
                                    "target": tgt,
                                    "relation": rel,
                                    "chunk_id": chunk_id
                                })
        except (json.JSONDecodeError, ValueError):
            pass

    _set_relation_cache(text, entities_str, relations)
    return relations


# ──────────────── Construction du graphe ────────────────

def build_entity_graph(docs: list, use_llm_relations: bool = True,
                       model: str = None,
                       progress_callback=None) -> nx.DiGraph:
    """
    Construit un graphe d'entités à partir d'une liste de Documents LangChain.

    Deux modes de construction :
    1) Co-occurrence (toujours) : si deux entités apparaissent dans le même chunk,
       elles sont liées par une arête "co_occurrence" avec un poids cumulé.
    2) LLM (optionnel) : le LLM extrait des relations typées (certifie, évalue, etc.)

    Args:
        docs: liste de Documents enrichis (avec entities_flat dans metadata)
        use_llm_relations: si True, appelle le LLM pour extraire des relations typées
        model: modèle Ollama pour l'extraction de relations
        progress_callback: callable(current, total)

    Returns:
        nx.DiGraph avec nœuds (entités) et arêtes (relations)
    """
    G = nx.DiGraph()

    total = len(docs)
    llm_relation_count = 0

    for i, doc in enumerate(docs):
        entities_dict = doc.metadata.get("entities", {})
        entities_flat = doc.metadata.get("entities_flat", [])
        chunk_id = doc.metadata.get("id", f"chunk_{i}")

        if not entities_flat:
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        # Ajouter les nœuds avec leur type
        for cat, ents in entities_dict.items():
            if isinstance(ents, list):
                for e in ents:
                    if G.has_node(e):
                        # Incrémenter le compteur d'apparitions
                        G.nodes[e]["count"] = G.nodes[e].get("count", 1) + 1
                        # Ajouter le chunk_id à la liste des chunks
                        G.nodes[e].setdefault("chunk_ids", set()).add(chunk_id)
                    else:
                        G.add_node(e, type=cat, count=1, chunk_ids={chunk_id})

        # ── Co-occurrences : relier toutes les paires d'entités du chunk ──
        # Extraire les noms bruts (sans préfixe TYPE:) depuis entities_dict
        raw_names = []
        for cat, ents in entities_dict.items():
            if isinstance(ents, list):
                raw_names.extend(ents)
        unique_entities = list(set(raw_names))

        for j in range(len(unique_entities)):
            for k in range(j + 1, len(unique_entities)):
                src, tgt = unique_entities[j], unique_entities[k]
                if G.has_edge(src, tgt):
                    G[src][tgt]["weight"] = G[src][tgt].get("weight", 1) + 1
                    G[src][tgt].setdefault("chunk_ids", set()).add(chunk_id)
                else:
                    G.add_edge(src, tgt, relation="co_occurrence",
                               weight=1, chunk_ids={chunk_id})
                # Arête inverse aussi (non-dirigée pour co-occurrence)
                if G.has_edge(tgt, src):
                    G[tgt][src]["weight"] = G[tgt][src].get("weight", 1) + 1
                    G[tgt][src].setdefault("chunk_ids", set()).add(chunk_id)
                else:
                    G.add_edge(tgt, src, relation="co_occurrence",
                               weight=1, chunk_ids={chunk_id})

        # ── Relations LLM (optionnel) ──
        if use_llm_relations and len(unique_entities) >= 2:
            relations = extract_relations_from_chunk(
                doc.page_content, entities_flat,
                chunk_id=chunk_id, model=model
            )
            for rel in relations:
                src = rel["source"]
                tgt = rel["target"]
                rel_type = rel["relation"]

                # S'assurer que les nœuds existent
                if not G.has_node(src):
                    G.add_node(src, type="UNKNOWN", count=1, chunk_ids={chunk_id})
                if not G.has_node(tgt):
                    G.add_node(tgt, type="UNKNOWN", count=1, chunk_ids={chunk_id})

                # Ajouter l'arête typée (peut écraser une co_occurrence)
                if G.has_edge(src, tgt) and G[src][tgt].get("relation") == "co_occurrence":
                    # Upgrader de co_occurrence vers relation typée
                    G[src][tgt]["relation"] = rel_type
                    G[src][tgt].setdefault("chunk_ids", set()).add(chunk_id)
                elif not G.has_edge(src, tgt):
                    G.add_edge(src, tgt, relation=rel_type,
                               weight=1, chunk_ids={chunk_id})

                llm_relation_count += 1

        if progress_callback:
            progress_callback(i + 1, total)

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"[graph] {i+1}/{total} chunks traités — "
                  f"{G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

    print(f"[graph] Graphe final : {G.number_of_nodes()} nœuds, "
          f"{G.number_of_edges()} arêtes "
          f"({llm_relation_count} relations LLM)")

    return G


# ──────────────── Persistance MongoDB ────────────────

def save_graph_to_mongo(G: nx.DiGraph, source_doc: str, db_name: str = "ragdb",
                        collection_name: str = "entity_graph"):
    """
    Sauvegarde le graphe dans MongoDB pour un document spécifique.
    Les sets sont convertis en listes pour la sérialisation.
    """
    client = MongoClient("mongodb://localhost:27017")
    db = client[db_name]
    col = db[collection_name]

    # Vider uniquement les données du document courant
    col.delete_many({"source_doc": source_doc})

    # Sauvegarder les nœuds
    nodes = []
    for node, data in G.nodes(data=True):
        node_data = dict(data)
        # Convertir les sets en listes
        if "chunk_ids" in node_data:
            node_data["chunk_ids"] = list(node_data["chunk_ids"])
        nodes.append({
            "_id": f"node:{source_doc}:{node}",
            "entity": node,
            "doc_type": "node",
            "source_doc": source_doc,
            **node_data
        })

    # Sauvegarder les arêtes
    edges = []
    for src, tgt, data in G.edges(data=True):
        edge_data = dict(data)
        if "chunk_ids" in edge_data:
            edge_data["chunk_ids"] = list(edge_data["chunk_ids"])
        edges.append({
            "_id": f"edge:{source_doc}:{src}→{tgt}",
            "source": src,
            "target": tgt,
            "doc_type": "edge",
            "source_doc": source_doc,
            **edge_data
        })

    if nodes:
        col.insert_many(nodes)
    if edges:
        col.insert_many(edges)

    print(f"[graph] Graphe sauvegardé dans MongoDB pour {source_doc} : "
          f"{len(nodes)} nœuds, {len(edges)} arêtes")


def load_graph_from_mongo(source_doc: str, db_name: str = "ragdb",
                          collection_name: str = "entity_graph") -> Optional[nx.DiGraph]:
    """
    Charge le graphe depuis MongoDB pour un document spécifique.
    Retourne None si aucun graphe n'est trouvé pour ce document.
    """
    client = MongoClient("mongodb://localhost:27017")
    db = client[db_name]
    col = db[collection_name]

    if col.count_documents({"source_doc": source_doc}) == 0:
        return None

    G = nx.DiGraph()

    # Charger les nœuds
    for doc in col.find({"doc_type": "node", "source_doc": source_doc}):
        entity = doc["entity"]
        attrs = {k: v for k, v in doc.items()
                 if k not in ("_id", "entity", "doc_type", "source_doc")}
        # Reconvertir les listes en sets
        if "chunk_ids" in attrs:
            attrs["chunk_ids"] = set(attrs["chunk_ids"])
        G.add_node(entity, **attrs)

    # Charger les arêtes
    for doc in col.find({"doc_type": "edge", "source_doc": source_doc}):
        src = doc["source"]
        tgt = doc["target"]
        attrs = {k: v for k, v in doc.items()
                 if k not in ("_id", "source", "target", "doc_type", "source_doc")}
        if "chunk_ids" in attrs:
            attrs["chunk_ids"] = set(attrs["chunk_ids"])
        G.add_edge(src, tgt, **attrs)

    print(f"[graph] Graphe chargé depuis MongoDB pour {source_doc} : "
          f"{G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

    return G


# ──────────────── Stats du graphe ────────────────

def graph_stats(G: nx.DiGraph) -> dict:
    """Retourne des statistiques sur le graphe."""
    if G is None or G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0}

    # Compter les types de nœuds
    type_counts = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "UNKNOWN")
        type_counts[t] = type_counts.get(t, 0) + 1

    # Compter les types de relations
    rel_counts = {}
    for _, _, data in G.edges(data=True):
        r = data.get("relation", "unknown")
        rel_counts[r] = rel_counts.get(r, 0) + 1

    # Top entités par nombre de connexions
    top_entities = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "node_types": type_counts,
        "relation_types": rel_counts,
        "top_entities": [(e, d) for e, d in top_entities],
    }
