# utils/debug_utils.py
# Fonctions utilitaires pour afficher et debugger les résultats de recherche (RAG)
from typing import Dict, Any, List

# Renvoie une version tronquée d'un texte (remplace les retours à la ligne et limite la longueur)
def _short(txt: str, n: int = 120) -> str:
    t = (txt or "").replace("\n", " ")
    return t[:n] + ("…" if len(t) > n else "")

# Affichage ultra lisible et compact pour chaque résultat (1 ligne par chunk)
def print_simple_results(title: str, items: List[Dict[str, Any]], max_items: int = 10, fields=None):
    print(f"\n=== {title} (top {min(len(items), max_items)}) ===")
    if not items:
        print("Aucun résultat.")
        return
    # Détermine les champs à afficher
    if fields is None:
        # Champs par défaut : score global, rrf, bm25, sim, ce, id, meta
        fields = [
            ("score_global", "Score"),
            ("bm25_norm", "BM25n"),
            ("sim_norm", "Simn"),
            ("rrf", "RRF"),
            ("ce_score", "CE"),
            ("bm25", "BM25"),
            ("sim_est", "Sim"),
            ("id", "ID"),
        ]
    # En-tête
    header = " | ".join([f"{label:>7}" for _, label in fields] + ["Résumé"])
    print(header)
    print("-" * len(header))
    # Affichage ligne par ligne
    for it in items[:max_items]:
        vals = []
        for key, _ in fields:
            v = it.get(key)
            if v is None:
                vals.append("   -   ")
            elif isinstance(v, float):
                vals.append(f"{v:7.3f}")
            else:
                vals.append(str(v)[:7].ljust(7))
        vals.append(_short(it.get("doc", "")))
        print(" | ".join(vals))

