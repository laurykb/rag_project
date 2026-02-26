# indexing/store_mongo.py
from pymongo import MongoClient

def save_chunks_to_mongo(docs, db_name="ragdb", collection_name="chunks"):
    client = MongoClient("mongodb://localhost:27017")
    db = client[db_name]
    col = db[collection_name]

    for doc in docs:
        record = {
            "_id": doc.metadata["id"],
            "content": doc.page_content,
            "chunk_idx": doc.metadata["chunk_idx"],
            "section_idx": doc.metadata["section_idx"],
            "source": doc.metadata["source"],
            "page_number": doc.metadata.get("page_number"),
            "chunk_type": doc.metadata.get("chunk_type", "chunk"),
            # Enrichissement LLM (vide si non activé)
            "keywords": doc.metadata.get("keywords", []),
            "keywords_str": doc.metadata.get("keywords_str", ""),
            "questions": doc.metadata.get("questions", []),
            "questions_str": doc.metadata.get("questions_str", ""),
            # RAPTOR metadata
            "heading": doc.metadata.get("heading", ""),
            "breadcrumb": doc.metadata.get("breadcrumb", ""),
            "summary_num_chunks": doc.metadata.get("summary_num_chunks"),
            # Étape 5 : tables/figures
            "table_description": doc.metadata.get("table_description", ""),
            "has_table": doc.metadata.get("has_table", False),
            "has_figure": doc.metadata.get("has_figure", False),
            # Étape 7 : entités nommées
            "entities": doc.metadata.get("entities", {}),
            "entities_flat": doc.metadata.get("entities_flat", []),
            "entities_str": doc.metadata.get("entities_str", ""),
        }

        col.update_one({"_id": record["_id"]}, {"$set": record}, upsert=True)

    print(f"{len(docs)} chunks enregistrés dans MongoDB.")

def save_query_to_mongo(query, db_name="ragdb", collection_name="queries"):
    client = MongoClient("mongodb://localhost:27017")
    db = client[db_name]
    col = db[collection_name]

    record = {
        "query": query,
    }

    col.insert_one(record)
    print(f"Requête enregistrée : {query}")

"""
COMMANDE DOCKER : 
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -v ~/mongodb_data:/data/db \
  mongo:7.0
"""