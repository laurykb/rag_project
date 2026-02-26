# SSH-GPT — Pipeline RAG

> Pipeline de Retrieval-Augmented Generation (RAG) hybride avec interface Streamlit.

## Architecture du projet

```
RAG_Laury/
│
├── config.py                        # Configuration centralisée (modèles, hyperparamètres)
│
├── app/                             # Interface utilisateur
│   └── streamlit_app.py             #     Application Streamlit (8 onglets)
│
├── core/                            # Logique métier de la pipeline
│   ├── ask.py                       #     Orchestrateur : query → rewrite → retrieve → réponse
│   ├── ingest.py                    #     Ingestion : markdown → chunks → embeddings → index
│   └── llm_answer.py                #     Construction du contexte + appel LLM (réponse finale)
│
├── retrieval/                       # Recherche et fusion
│   ├── retrieve.py                  #     Orchestrateur du retrieval hybride
│   ├── semantic_search.py           #     Recherche sémantique via ChromaDB
│   ├── keyword_bm25.py              #     Recherche par mots-clés (BM25)
│   ├── rrf.py                       #     Reciprocal Rank Fusion (fusion de classements)
│   └── cross_encoder.py             #     Reranking final avec Cross-Encoder
│
├── indexing/                        # Chunking, embedding et stockage
│   ├── chunking.py                  #     Découpe sémantique du markdown en chunks
│   ├── embedding.py                 #     Génération d'embeddings + indexation ChromaDB
│   ├── keyword_index.py             #     Construction de l'index BM25
│   └── store_mongo.py               #     Stockage des chunks dans MongoDB
│
├── nlp/                             # Traitement du langage naturel
│   ├── query_rewriter.py            #     Réécriture de requête (guarded rewrite)
│   ├── vocab_builder.py             #     Extraction de vocabulaire et acronymes
│   └── ollama_embedding.py          #     Wrapper pour les embeddings Ollama
│
├── preprocessing/                   # Conversion de documents
│   └── pdf_to_markdown.py           #     Conversion PDF → Markdown (Docling)
│
├── utils/                           # Utilitaires transversaux
│   ├── text_utils.py                #     normalize_text, make_doc_id
│   └── debug_utils.py               #     Affichage et debug des résultats
│
├── data/                            # Données générées (runtime)
│   ├── chroma_db/                   #     Base vectorielle ChromaDB
│   ├── vocab_save/                  #     Vocabulaire et acronymes (vocab.json)
│   ├── out_clean_md/                #     Markdown nettoyés (prêts à ingérer)
│   └── bm25_index.pkl               #     Index BM25 sérialisé
│
├── models/                          # Modèles locaux
│   └── ms-marco-MiniLM-L-6-v2/     #     Cross-Encoder pour le reranking
│
├── docs/                            # Documentation et fichiers Docling bruts
│   └── out/                         #     Markdown brut (sortie Docling)
│
├── rag_venv/                        # Environnement virtuel Python
└── useless_src/                     # Fichiers archivés (non utilisés)
```

## Pipeline de traitement

```
                ┌──────────────┐
                │   PDF source │
                └──────┬───────┘
                       │  preprocessing/pdf_to_markdown.py
                       ▼
                ┌──────────────┐
                │   Markdown   │
                └──────┬───────┘
                       │  core/ingest.py
                       ▼
        ┌──────────────────────────────┐
        │  indexing/chunking.py        │  Découpe sémantique
        │  indexing/embedding.py       │  Embeddings → ChromaDB
        │  indexing/keyword_index.py   │  Index BM25
        │  indexing/store_mongo.py     │  Stockage MongoDB
        │  nlp/vocab_builder.py        │  Vocabulaire / acronymes
        └──────────────────────────────┘
```

```
        ┌──────────────────┐
        │  Question user   │
        └────────┬─────────┘
                 │  core/ask.py
                 ▼
        ┌──────────────────┐
        │  Query Rewrite   │  nlp/query_rewriter.py
        └────────┬─────────┘
                 ▼
    ┌────────────┴────────────┐
    │                         │
    ▼                         ▼
┌────────┐             ┌──────────┐
│Semantic│             │  BM25    │
│ Search │             │  Search  │
└───┬────┘             └────┬─────┘
    │                       │
    └───────────┬───────────┘
                ▼
        ┌──────────────┐
        │  RRF Fusion  │  retrieval/rrf.py
        └──────┬───────┘
               ▼
        ┌──────────────┐
        │  Cross-Enc.  │  retrieval/cross_encoder.py
        │  Reranking   │
        └──────┬───────┘
               ▼
        ┌──────────────┐
        │  LLM Answer  │  core/llm_answer.py
        └──────────────┘
```

## Lancement

```bash
# Activer l'environnement virtuel
source rag_venv/bin/activate

# Lancer l'interface Streamlit
streamlit run app/streamlit_app.py
```

## Technologies

| Composant | Technologie |
|-----------|------------|
| LLM / Embeddings | Ollama (local) |
| Base vectorielle | ChromaDB |
| Recherche BM25 | rank_bm25 |
| Reranking | Cross-Encoder (sentence-transformers) |
| Stockage chunks | MongoDB |
| Interface | Streamlit |
| Conversion PDF | Docling |
