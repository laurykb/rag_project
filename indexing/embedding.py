# indexing/embedding.py
# Fonctions pour générer les embeddings de documents et indexer dans une base vectorielle ChromaDB
from nlp.ollama_embedding import OllamaEmbedding
import chromadb
from config import CHROMA_PATH, COLLECTION_NAME


def build_embeddings(docs):
    """
    Génère les embeddings (vecteurs) pour une liste de documents langchain à l'aide du modèle OllamaEmbedding.
    
    Stratégie d'embedding (inspirée RAGFlow) :
    - Si le chunk a des questions générées (auto_questions), l'embedding est calculé
      sur les questions plutôt que le contenu brut, car question↔question matching
      est un signal de pertinence plus fort que contenu↔question.
    - Sinon, fallback sur le contenu brut.
    
    Retourne les textes, les vecteurs, les métadonnées et les identifiants associés à chaque chunk.

    Args:
        docs (list): Liste d'objets Document (doivent avoir .page_content et .metadata)

    Returns:
        tuple: (texts, vecs, metadatas, ids)
            - texts (list[str]): Textes des chunks
            - vecs (list[list[float]]): Embeddings vectoriels
            - metadatas (list[dict]): Métadonnées associées à chaque chunk
            - ids (list[str]): Identifiants uniques de chaque chunk
    """
    # Instanciation du modèle d'embedding (Ollama)
    model = OllamaEmbedding()
    
    # Filtrer les documents vides (pas de texte = pas d'embedding utile)
    docs = [d for d in docs if d.page_content.strip()]
    
    # Extraction du texte de chaque chunk
    texts = [d.page_content.strip() for d in docs]
    
    # Textes à embedder : questions si disponibles, sinon contenu brut
    texts_to_embed = []
    for d in docs:
        questions_str = d.metadata.get("questions_str", "")
        if questions_str and len(questions_str) > 20:
            # Embedding sur les questions générées (meilleur pour le retrieval)
            texts_to_embed.append(questions_str)
        else:
            texts_to_embed.append(d.page_content.strip())
    
    # Calcul des embeddings
    vecs = model.embed_documents(texts_to_embed)
    
    # Métadonnées : on sérialise les listes en strings pour ChromaDB
    metadatas = []
    for d in docs:
        meta = dict(d.metadata)  # copie
        # ChromaDB n'accepte pas les listes/dicts dans metadata, on les sérialise
        if isinstance(meta.get("keywords"), list):
            meta["keywords"] = ", ".join(meta["keywords"])
        if isinstance(meta.get("questions"), list):
            meta["questions"] = " | ".join(meta["questions"])
        # Étape 7 : entités — sérialiser en strings pour ChromaDB
        if isinstance(meta.get("entities"), dict):
            from nlp.ner_extractor import entities_to_str
            meta["entities"] = entities_to_str(meta["entities"])
        if isinstance(meta.get("entities_flat"), list):
            meta["entities_flat"] = ", ".join(meta["entities_flat"])
        metadatas.append(meta)
    
    # Récupération des identifiants uniques
    ids = [d.metadata["id"] for d in docs]
    return texts, vecs, metadatas, ids


def index_chroma(ids, texts, metadatas, embeddings, collection_name=COLLECTION_NAME, clean_collection=True):
    """
    Indexe les embeddings, textes, ids et métadonnées dans une base vectorielle ChromaDB persistante.
    Si clean_collection=True, supprime la collection avant d'ajouter (évite les doublons).

    Args:
        ids (list[str]): Identifiants uniques des chunks
        texts (list[str]): Textes des chunks
        metadatas (list[dict]): Métadonnées associées à chaque chunk
        embeddings (list[list[float]]): Embeddings vectoriels
        collection_name (str): Nom de la collection ChromaDB (défaut: COLLECTION_NAME)
        clean_collection (bool): Indique si la collection doit être nettoyée avant l'indexation (défaut: True)

    Returns:
        collection: Instance de la collection ChromaDB contenant les données indexées
    """
    # Connexion à la base ChromaDB persistante (chemin défini dans config)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    if clean_collection:
        try:
            # Suppression de la collection existante pour éviter les doublons
            client.delete_collection(collection_name)
        except Exception:
            pass  # La collection n'existait pas, pas de souci
    # Création ou récupération de la collection ChromaDB avec configuration HNSW (recherche par similarité cosinus)
    collection = client.get_or_create_collection(
        name=collection_name,
        configuration={
            "hnsw": {
                "space": "cosine",  # Métrique de similarité
                "ef_construction": 200, 
                "ef_search": 50
            }
        },
        embedding_function=None  # Les embeddings sont fournis, pas calculés par ChromaDB
    )

    # Ajout des données (ids, textes, métadonnées, embeddings) dans la collection
    collection.add(
        ids=ids, 
        documents=texts, 
        metadatas=metadatas, 
        embeddings=embeddings
    )
    return collection




