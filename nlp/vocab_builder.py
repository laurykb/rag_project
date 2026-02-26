# nlp/vocab_builder.py
# Outils pour construire et gérer un vocabulaire de mots fréquents et d'acronymes à partir d'une liste de documents.
import re
import json
from pathlib import Path
from collections import Counter
from langchain_core.documents import Document

# Expression régulière pour extraire les mots et acronymes (mots, tirets, apostrophes)
TOKEN_RE = re.compile(r"\b[\w\-’']+\b", flags=re.UNICODE)


def build_vocab(docs, top_k_terms=3000):
    """
    Construit un vocabulaire de mots fréquents et d'acronymes à partir d'une liste de documents.
    - Parcourt chaque document et extrait tous les mots.
    - Ajoute les acronymes (suite de lettres majuscules, 2 à 10 caractères) dans un ensemble dédié.
    - Compte la fréquence de chaque mot (en minuscules) et garde les top_k_terms plus fréquents.
    Args:
        docs (list): Liste d'objets Document (doivent avoir .page_content)
        top_k_terms (int): Nombre maximum de mots fréquents à retenir
    Returns:
        tuple: (vocabulaire, acronymes) sous forme d'ensembles
    """
    mots = []  # Liste de tous les mots rencontrés
    acronymes = set()  # Ensemble des acronymes détectés
    for doc in docs:
        texte = getattr(doc, 'page_content', '') or ''  # Récupère le texte du document
        for mot in TOKEN_RE.findall(texte):  # Pour chaque mot extrait
            if mot.isupper() and mot.isalpha() and 2 <= len(mot) <= 10:
                acronymes.add(mot)  # Ajoute l'acronyme si critère rempli
            mots.append(mot.lower())  # Ajoute le mot en minuscule
    # On garde les top_k_terms mots les plus fréquents
    vocabulaire = set([mot for mot, _ in Counter(mots).most_common(top_k_terms)])
    return vocabulaire, acronymes


def save_vocab(vocabulaire, acronymes, chemin="/home/marsattacks/Documents/RAG_Laury/data/vocab_save/vocab.json", path=None):
    """
    Sauvegarde le vocabulaire et les acronymes dans un fichier JSON.
    Args:
        vocabulaire (set): Ensemble des mots fréquents
        acronymes (set): Ensemble des acronymes
        chemin (str): Chemin du fichier de sortie (défaut: vocab.json)
        path (str): Alias 
    """
    if chemin is None and path is not None:
        chemin = path
    if chemin is None:
        chemin = "vocab.json"
    data = {
        "vocab_terms": sorted(vocabulaire),
        "acronyms": sorted(acronymes)
    }
    return Path(chemin).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vocab(chemin="/home/marsattacks/Documents/RAG_Laury/data/vocab_save/vocab.json"):
    """
    Charge le vocabulaire et les acronymes depuis un fichier JSON.
    Args:
        chemin (str): Chemin du fichier JSON à charger
    Returns:
        tuple: (vocabulaire, acronymes) sous forme d'ensembles
    """
    data = json.loads(Path(chemin).read_text(encoding="utf-8"))
    return set(data.get("vocab_terms", [])), set(data.get("acronyms", []))
