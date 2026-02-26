from langchain_core.embeddings import Embeddings
import requests
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Cache LRU process-level : {(model, hash(text)) -> vector} ─────────────────
# Évite de rappeler Ollama pour une requête déjà embeddée dans la même session.
# Taille max : 256 entrées (~256 × 1024 floats × 4 bytes ≈ 1 MB — négligeable)
_EMBED_CACHE: dict = {}
_EMBED_CACHE_MAX = 256

def _cache_key(model: str, text: str) -> str:
    h = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
    return f"{model}:{h}"


class OllamaEmbedding(Embeddings):
    def __init__(self, model="bge-m3:567m", base_url="http://localhost:11434", max_workers: int = 4):
        self.model = model
        self.base_url = base_url
        self.max_workers = max_workers  # parallélisme pour embed_documents
        self._dim = None  # dimension auto-detectee au premier appel reussi

    def embed_documents(self, texts):
        """Encode plusieurs textes en parallèle (max_workers threads simultanés)."""
        results = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.embed_query, text): i for i, text in enumerate(texts)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def embed_query(self, text):
        # Proteger contre les textes vides (Ollama retourne 500)
        if not text or not text.strip():
            return self._zero_vector()

        # ── Cache hit : retour immédiat sans appel réseau ──────────────────────
        key = _cache_key(self.model, text)
        if key in _EMBED_CACHE:
            return _EMBED_CACHE[key]

        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            vec = response.json()["embedding"]
            # Memoriser la dimension au premier succes
            if self._dim is None and vec:
                self._dim = len(vec)
            # ── Stocker dans le cache (avec LRU basique par FIFO si plein) ────
            if len(_EMBED_CACHE) >= _EMBED_CACHE_MAX:
                # Retire le premier élément inséré (FIFO)
                oldest = next(iter(_EMBED_CACHE))
                del _EMBED_CACHE[oldest]
            _EMBED_CACHE[key] = vec
            return vec
        except Exception as e:
            print(f"Erreur lors de l'appel à Ollama : {e}")
            return self._zero_vector()

    def _zero_vector(self):
        """Vecteur zero de la bonne dimension (auto-detectee ou 1024 par defaut)."""
        dim = self._dim if self._dim else 1024
        return [0.0] * dim
