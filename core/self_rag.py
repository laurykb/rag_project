# core/self_rag.py
"""
Self-RAG — pipeline RAG avec auto-évaluation et retry.

Principe :
  1. Retrieval + génération classique (via ask._prepare_retrieval + llm_answer.answer)
  2. Évaluation automatique des 3 métriques LLM-as-judge :
       - context_relevance : les chunks sont-ils pertinents ?  (qualité du retrieval)
       - faithfulness       : la réponse est-elle fondée ?      (pas d'hallucination)
       - answer_relevance   : la réponse répond-elle à la question ?
  3. Si le score global est insuffisant (< SELF_RAG_THRESHOLD) ET qu'il reste des tentatives :
       - On reformule la requête différemment (stratégie de rewrite alternatif)
       - On recommence le retrieval+génération (max MAX_RETRIES fois)
  4. On retourne la MEILLEURE réponse parmi les tentatives (score le plus élevé).

Paramètres configurables dans config.py :
  SELF_RAG_ENABLED    : bool  — active/désactive le Self-RAG (défaut : False)
  SELF_RAG_THRESHOLD  : float — seuil de score moyen sous lequel on retente (défaut : 0.55)
  SELF_RAG_MAX_RETRIES: int   — nombre maximum de tentatives supplémentaires (défaut : 1)

Intégration :
  Utilisé par process_query() et process_query_stream() dans ask.py
  lorsque SELF_RAG_ENABLED = True.
"""

from __future__ import annotations

from typing import Optional

from config import (
    REWRITER_MODEL,
    SELF_RAG_ENABLED,
    SELF_RAG_THRESHOLD,
    SELF_RAG_MAX_RETRIES,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Score global : moyenne pondérée des 3 métriques
# ─────────────────────────────────────────────────────────────────────────────

_WEIGHTS = {
    "context_relevance": 0.30,   # qualité du retrieval
    "faithfulness":       0.45,   # pas d'hallucination — critère le plus important
    "answer_relevance":   0.25,   # utilité de la réponse
}


def _weighted_score(metrics: dict) -> float:
    """Calcule un score global pondéré à partir des 3 métriques LLM-as-judge."""
    total_w = 0.0
    total   = 0.0
    for key, w in _WEIGHTS.items():
        val = metrics.get(key)
        if val is not None:
            total   += val * w
            total_w += w
    return total / total_w if total_w > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Reformulation alternative pour le retry
# ─────────────────────────────────────────────────────────────────────────────

def _alternative_rewrite(question: str, attempt: int, llm) -> str:
    """
    Génère une reformulation alternative de la question pour le retry.
    Chaque tentative utilise une consigne différente pour diversifier le retrieval.
    """
    strategies = [
        # tentative 1 : reformulation plus technique et précise
        f"""Reformule la question suivante de façon plus technique et précise,
en utilisant des termes spécialisés du domaine. Réponds UNIQUEMENT avec la question reformulée.

Question originale : {question}
Question reformulée :""",

        # tentative 2 : décomposition / aspect différent
        f"""La question suivante n'a pas obtenu une bonne réponse. Reformule-la en te
concentrant sur un aspect plus spécifique ou en la décomposant différemment.
Réponds UNIQUEMENT avec la question reformulée.

Question originale : {question}
Question reformulée :""",
    ]

    strategy = strategies[min(attempt - 1, len(strategies) - 1)]
    try:
        result = llm.invoke(strategy).strip()
        if result and len(result) > 5 and result != question:
            print(f"[Self-RAG] Tentative {attempt} — reformulation : '{result}'")
            return result
    except Exception as e:
        print(f"[Self-RAG] Erreur reformulation alternative : {e}")
    return question


# ─────────────────────────────────────────────────────────────────────────────
#  Évaluation rapide d'une tentative
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_attempt(
    question: str,
    answer: str,
    chunks: list,
    judge_llm,
) -> tuple[dict, float]:
    """
    Lance les 3 métriques LLM-as-judge sur une tentative.
    Retourne (metrics_dict, weighted_score).
    """
    from core.evaluation import faithfulness_llm, answer_relevance_llm, context_relevance_llm

    metrics = {
        "faithfulness":       faithfulness_llm(answer, chunks, llm=judge_llm),
        "answer_relevance":   answer_relevance_llm(answer, question, llm=judge_llm),
        "context_relevance":  context_relevance_llm(chunks, question, llm=judge_llm),
    }
    score = _weighted_score(metrics)
    print(
        f"[Self-RAG] Score : {score:.2f}  "
        f"(faith={metrics['faithfulness']:.2f}, "
        f"ans_rel={metrics['answer_relevance']:.2f}, "
        f"ctx_rel={metrics['context_relevance']:.2f})"
    )
    return metrics, score


# ─────────────────────────────────────────────────────────────────────────────
#  Point d'entrée principal — non-streaming
# ─────────────────────────────────────────────────────────────────────────────

def self_rag_query(
    user_q: str,
    system_prompt: Optional[str] = None,
    source_filter: Optional[str] = None,
    conversation_history: Optional[list] = None,
) -> tuple[str, list, list, dict]:
    """
    Exécute le pipeline Self-RAG complet (non-streaming).

    Retourne : (best_answer, best_chunks, best_citations, best_metrics)
    """
    from langchain_community.llms import Ollama
    from core.ask import _prepare_retrieval
    from core.llm_answer import answer as llm_answer
    from core.evaluation import _get_judge_llm

    judge_llm  = _get_judge_llm()
    llm_writer = Ollama(model=REWRITER_MODEL, temperature=0.2)

    best_answer    = ""
    best_chunks    = []
    best_citations = []
    best_metrics   = {}
    best_score     = -1.0

    current_q = user_q

    for attempt in range(SELF_RAG_MAX_RETRIES + 1):
        print(f"\n[Self-RAG] ── Tentative {attempt + 1}/{SELF_RAG_MAX_RETRIES + 1} ──")

        # ── Retrieval ─────────────────────────────────────────────────────────
        q_main, chunks = _prepare_retrieval(
            current_q,
            source_filter=source_filter,
            conversation_history=conversation_history,
        )

        if not chunks:
            print("[Self-RAG] Aucun chunk trouvé, arrêt.")
            break

        # ── Génération ────────────────────────────────────────────────────────
        response, citations = llm_answer(
            q_main, chunks,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
        )

        # ── Évaluation ────────────────────────────────────────────────────────
        metrics, score = _evaluate_attempt(user_q, response, chunks, judge_llm)

        # Garde la meilleure tentative
        if score > best_score:
            best_score     = score
            best_answer    = response
            best_chunks    = chunks
            best_citations = citations
            best_metrics   = metrics

        # Seuil atteint → inutile de continuer
        if score >= SELF_RAG_THRESHOLD:
            print(f"[Self-RAG] Seuil atteint ({score:.2f} >= {SELF_RAG_THRESHOLD}) — réponse acceptée.")
            break

        # Dernier essai échoué → on garde quand même la meilleure réponse
        if attempt >= SELF_RAG_MAX_RETRIES:
            print(
                f"[Self-RAG] Score insuffisant ({best_score:.2f} < {SELF_RAG_THRESHOLD}) "
                f"après {attempt + 1} tentative(s) — on renvoie la meilleure réponse disponible."
            )
            break

        # Reformulation alternative pour le prochain essai
        current_q = _alternative_rewrite(user_q, attempt + 1, llm_writer)

    best_metrics["self_rag_score"]    = best_score
    best_metrics["self_rag_attempts"] = attempt + 1  # noqa: F821 (toujours défini après la boucle)
    return best_answer, best_chunks, best_citations, best_metrics
