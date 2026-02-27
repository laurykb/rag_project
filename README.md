# SSH-GPT — Pipeline RAG

> Pipeline de Retrieval-Augmented Generation (RAG) hybride avec interface Streamlit.  
> Documents cibles : PDFs techniques (Critères Communs, ANSSI, référentiels de sécurité).

---

## Architecture du projet

```
RAG_Laury/
│
├── config.py                        # Configuration centralisée (modèles, hyperparamètres, seuils)
│
├── app/
│   └── streamlit_app.py             # Interface Streamlit multi-onglets
│
├── core/
│   ├── ask.py                       # Orchestrateur : query → rewrite → retrieve → réponse
│   │                                #   + get_docs_with_graph() (disponibilité GraphRAG par doc)
│   ├── ingest.py                    # Ingestion : markdown → chunks → embeddings → index
│   └── llm_answer.py                # Construction du contexte + appel LLM
│
├── retrieval/
│   ├── retrieve.py                  # Orchestrateur retrieval hybride (sem + BM25 + graph)
│   ├── semantic_search.py           # Recherche sémantique via ChromaDB
│   ├── keyword_bm25.py              # Recherche par mots-clés (BM25)
│   ├── rrf.py                       # Reciprocal Rank Fusion
│   ├── cross_encoder.py             # Reranking final (bge-reranker-v2-m3, cuda:1)
│   └── graph_retrieve.py            # Retrieval via graphe d'entités (GraphRAG)
│
├── indexing/
│   ├── chunking.py                  # Découpe sémantique + Parent-Child + déduplication
│   ├── embedding.py                 # Embeddings → ChromaDB
│   ├── keyword_index.py             # Index BM25
│   └── store_mongo.py               # Stockage MongoDB (ragdb.chunks)
│
├── nlp/
│   ├── query_rewriter.py            # Réécriture de requête (guarded rewrite)
│   ├── vocab_builder.py             # Extraction vocabulaire / acronymes
│   └── ollama_embedding.py          # Wrapper embeddings Ollama
│
├── preprocessing/
│   └── pdf_to_markdown.py           # Conversion PDF → Markdown (Docling)
│
├── utils/
│   ├── text_utils.py                # normalize_text, make_doc_id
│   └── debug_utils.py               # Affichage debug résultats
│
├── data/
│   ├── chroma_db/                   # Base vectorielle ChromaDB
│   ├── vocab_save/                  # Vocabulaire sérialisé (vocab.json)
│   ├── out_clean_md/                # Markdown nettoyés (prêts à ingérer)
│   └── bm25_index.pkl               # Index BM25 sérialisé
│
├── models/
│   └── bge-reranker-v2-m3/          # Cross-Encoder local (BAAI, cuda:1)
│
└── rag_venv/                        # Environnement virtuel Python
```

---

## Pipeline

### Ingestion

```
PDF  →  Docling  →  Markdown  →  core/ingest.py
                                      │
              ┌───────────────────────┼────────────────────────────┐
              ▼                       ▼                            ▼
    Chunking sémantique        Index BM25               Graphe d'entités (opt.)
    Parent-Child               keyword_index.py         ragdb.entity_graph
    Déduplication (fingerprint + ID)
    Embeddings → ChromaDB
    Stockage → ragdb.chunks
    Vocabulaire / acronymes
```

### Requête

```
Question utilisateur
       │
       ▼  core/ask.py
       │
       ├── Condensation historique (multi-tours)
       ├── Query Rewrite [optionnel]  →  nlp/query_rewriter.py
       │
       ├── Détection hors-scope (CE < 0.51)  →  message refus si hors périmètre
       │
       ├── Recherche parallèle ──────────────────────────────────────┐
       │       Sémantique (ChromaDB)                                  │
       │       BM25                                                   │
       │       GraphRAG 🕸 [si doc a un graphe ET option activée]     │
       │                                                              │
       ▼                                                              │
   RRF Fusion  ◄────────────────────────────────────────────────────┘
       │
       ▼
   Cross-Encoder Reranking  (bge-reranker-v2-m3, cuda:1)
       │
       ▼
   build_context()  →  déduplication des textes (fingerprint 256 car.)
       │
       ▼
   LLM  (magistral:latest, cuda:0)
       │
       ▼
   Réponse streamée + citations
```

---

## Options pipeline (UI Streamlit)

| Option | Description | Défaut |
|--------|-------------|--------|
| **Réécriture de requête** | Reformule la question pour améliorer le rappel | OFF |
| **Parent-Child** | Remonte au chunk parent pour plus de contexte | ON |
| **GraphRAG 🕸** | Enrichit le retrieval via le graphe d'entités nommées | Auto |

> **GraphRAG** : le sélecteur de document affiche le badge 🕸 pour les docs qui ont un graphe dans MongoDB (`ragdb.entity_graph`). L'option est désactivée automatiquement si le document actif n'a pas de graphe.

---

## Détection hors-scope

Le score Cross-Encoder du meilleur chunk est comparé au seuil `CE_RELEVANCE_THRESHOLD = 0.51` (configurable dans `config.py`). En dessous du seuil, la requête est considérée hors périmètre du document — un message de refus est renvoyé sans appel LLM.

---

## Modèles

| Rôle | Modèle | Infra |
|------|--------|-------|
| Embeddings | `bge-m3:567m` (Ollama) | cuda:0 |
| Réécriture requête | `llama3.1` (Ollama) | cuda:0 |
| Génération réponse | `magistral:latest` (Ollama) | cuda:0 |
| Reranking | `bge-reranker-v2-m3` (local) | cuda:1 |
| NER (GraphRAG) | `fr_core_news_sm` (spaCy) | CPU |

---

## Technologies

| Composant | Technologie |
|-----------|-------------|
| LLM / Embeddings | Ollama (local) |
| Base vectorielle | ChromaDB |
| Recherche BM25 | rank_bm25 |
| Reranking | Cross-Encoder (sentence-transformers) |
| Stockage chunks + graphes | MongoDB (`ragdb`) |
| Interface | Streamlit |
| Conversion PDF | Docling |
| NER | spaCy `fr_core_news_sm` |

---

## Lancement

```bash
source rag_venv/bin/activate
streamlit run app/streamlit_app.py
```

Prérequis : Ollama en cours d'exécution, MongoDB actif sur `localhost:27017`.
