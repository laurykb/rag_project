# config.py
"""
Fichier de configuration central pour la pipeline RAG.
On regroupe ici les paramètres (modèles, topK, noms de collections...).
"""

from pathlib import Path
# Racine absolue du projet (répertoire contenant ce fichier config.py)
_PROJECT_ROOT = Path(__file__).resolve().parent

# ------------------- COLLECTION CHROMA DB -------------------
# Chemin vers la base de données Chroma utilisée pour le stockage des embeddings
CHROMA_PATH = str(_PROJECT_ROOT / "data" / "chroma_db")
# Nom de la collection ChromaDB utilisée pour l'indexation et la recherche
COLLECTION_NAME = "test_rag"

# ------------------ EMBEDDINGS ------------------
# Nom du modèle d'embedding utilisé via ton wrapper OllamaEmbedding
EMBED_MODEL = "bge-m3:567m"

# ------------------ LLMs ------------------
# Modèle LLM utilisé pour reformuler ou expanser les requêtes utilisateur
REWRITER_MODEL = "llama3.1:latest"
# Modèle LLM utilisé pour générer la réponse finale à l'utilisateur
GEN_MODEL = "magistral:latest"

# ------------------ Retrieval ------------------
# Nombre de chunks/documents utilisés dans la recherche (semantic + bm25) 
NUM_CHUNKS = 15

# Paramètre k pour la Reciprocal Rank Fusion (plus k est élevé, plus les scores sont lissés)
RRF_K = 60

# ------------------ Reranking ------------------
# Active ou désactive le reranking avec un cross-encoder (True = rerank activé)
USE_CROSS_ENCODER = True  # mets False pour désactiver le rerank
# Chemin local vers le modèle cross-encoder (doit contenir config.json, model.safetensors, tokenizer.*)
# bge-reranker-v2-m3 : modèle multilingue BAAI, supporte le français et l'anglais
CROSS_ENCODER_LOCAL_PATH = "/home/marsattacks/Documents/RAG_Laury/models/bge-reranker-v2-m3"
# GPU 1 réservé au cross-encoder 
CE_DEVICE = "cuda:1"

# ------------------ Divers ------------------
# Longueur max d’un contexte concaténé (avant envoi au LLM)
MAX_CHUNK_LENGTH = 25000  
# Longueur maximale (en caractères) d’une query réécrite/expansée
MAX_QUERY_CHARS = 512
# Nombre de paraphrases créées à partir du prompt utilisateur (0 = pas d’expansion)
N_EXPANSIONS = 0

WEIGHT_SEMANTIC = 0.3
WEIGHT_BM25 = 0.7

# --------------- ENRICHISSEMENT CHUNKS ---------------
# Active l'extraction automatique de mots-clés par chunk via LLM (0 = désactivé)
AUTO_KEYWORDS = 5
# Active la génération automatique de questions par chunk via LLM (0 = désactivé)
AUTO_QUESTIONS = 3
# Modèle utilisé pour l'enrichissement (None = utilise REWRITER_MODEL, plus fiable que les modèles thinking)
ENHANCEMENT_MODEL = None

# --------------- CHUNKING ---------------
# Mode de chunking : "naive" (split par headers) ou "technical" (hiérarchie numérotée, fusion parent-enfant)
CHUNKING_MODE = "technical"

# --------------- RAPTOR SIMPLIFIÉ ---------------
# Active la génération de résumés par section (chunks de type "summary" indexés avec les chunks normaux)
RAPTOR_SUMMARIES = True
# Nombre minimum de chunks dans une section pour déclencher la génération d'un résumé
RAPTOR_MIN_CHUNKS = 3
# Nombre max de chunks concaténés pour construire le prompt de résumé (éviter le dépassement de contexte)
RAPTOR_MAX_INPUT_CHUNKS = 15
#---------------------------------------------

# --------------- SELF-RAG ---------------
# Active le Self-RAG : évaluation automatique + retry si la réponse est insuffisante
# /!\ Coût : 3 appels LLM supplémentaires par tentative (évaluation) — désactiver si latence critique
SELF_RAG_ENABLED = False   # True = activé, False = pipeline classique (défaut)
# Score moyen (0.0-1.0) en dessous duquel on retente le retrieval+génération
SELF_RAG_THRESHOLD = 0.55
# Nombre maximum de tentatives supplémentaires (1 = 2 essais au total : initial + 1 retry)
SELF_RAG_MAX_RETRIES = 1
#---------------------------------------------

# --------------- PARENT-CHILD RETRIEVAL ---------------
# Active le Parent-Child (Small-to-Big) : après le rerank, remplace le contenu
# du chunk enfant par la section parente complète avant envoi au LLM.
# Améliore la richesse du contexte sans dégrader la précision du retrieval.
PARENT_CHILD_ENABLED   = False  # désactivé par défaut (latence) — activable via UI
# Taille maximale (en caractères) du texte parent concaténé envoyé au LLM par chunk.
# 8000 = contexte très riche mais prompt long (~120K avant troncature avec 15 chunks).
# 4000 = bon compromis richesse/vitesse (~2×plus rapide au LLM).
PARENT_CHILD_MAX_CHARS = 4000
# Nombre de chunks retenus quand Parent-Child est actif.
# Chaque chunk enfant est élargi à ~4000 chars : 8 chunks → ~32K de contexte utile,
# ce qui reste dans la fenêtre de qwen3-vl:8b sans saturation.
# (NUM_CHUNKS normal = 15 est utilisé quand Parent-Child est désactivé.)
NUM_CHUNKS_PARENT_CHILD = 8
#---------------------------------------------

# --------------- MONGO ---------------
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB  = "ragdb"
#---------------------------------------------

# --------------- LLM CONTEXT WINDOW ---------------
# Taille de la fenêtre de contexte envoyée à Ollama (en tokens).
# Défaut Ollama sans cette option : souvent 32768 → lent + forte conso VRAM.
# 16384 = bon compromis : couvre 15 chunks × ~800 tokens + prompt système.
# 24576 = pour Parent-Child (contexte parent ~4000 chars/chunk → plus grand).
LLM_NUM_CTX = 16384
#---------------------------------------------