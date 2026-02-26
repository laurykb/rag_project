import hashlib
# - supprime les espaces superflus, tabulations et retours multiples
# - retire les espaces en début/fin et réduit tous les espaces à un seul

def normalize_text(t: str) -> str:
    """
    Nettoie un texte pour le rendre plus stable :
    - supprime espaces en trop
    - supprime tabulations, retours multiples
    Args:
        t (str): Texte à normaliser
    Returns:
        str: Texte nettoyé et normalisé
    """
    return " ".join((t or "").strip().split())


# Génère un identifiant unique et stable pour un chunk de document
# Cela permet de retrouver ou comparer facilement les chunks, même si l'ordre change

def make_doc_id(text: str, source: str, chunk_idx: int | None) -> str:
    """
    Fabrique un identifiant unique et stable pour un chunk.
    - basé sur le texte normalisé
    - la source (nom du fichier)
    - l'index du chunk
    Args:
        text (str): Contenu du chunk
        source (str): Nom du fichier source
        chunk_idx (int | None): Index du chunk dans le document
    Returns:
        str: Identifiant unique (hash hexadécimal, 16 caractères)
    """
    txt = normalize_text(text)
    parts = []
    if source:
        parts.append(f"source={source}")
    if chunk_idx is not None:
        # Accepte int ou str, formate proprement
        if isinstance(chunk_idx, int):
            parts.append(f"idx={chunk_idx:06d}")
        else:
            parts.append(f"idx={str(chunk_idx)}")
    parts.append(f"text={txt}")

    key = "\n---\n".join(parts)
    # SHA256 pour robustesse, tronqué à 16 caractères
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

