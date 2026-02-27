from pathlib import Path
import sys
import os
import threading

# Ajouter la racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from pymongo import MongoClient
import re
from indexing.store_mongo import save_query_to_mongo
from core.ask import process_query, process_query_stream
from core.llm_answer import get_system_prompt, DEFAULT_SYSTEM_PROMPT
from core.chat_sessions import (
    create_session, get_session, list_sessions,
    add_message, delete_session, clear_session_messages,
    get_messages, update_session_source,
)
from core.ingest import ingest_markdown
from preprocessing.pdf_to_markdown import pdf_to_md, clean_md, convert_and_clean
from config import AUTO_KEYWORDS, AUTO_QUESTIONS, CHUNKING_MODE, RAPTOR_SUMMARIES
from indexing.chunking import CHUNKING_MODES


# ─────────────── Helper : affichage unifié d'un chunk ───────────────
def _chunk_display_label(meta: dict, rank: int = None, score: float = None) -> str:
    """
    Construit un label lisible pour un chunk à partir de ses métadonnées.
    Utilisé dans tab2 (Classement) et tab3 (Exploration).
    
    Exemples :
      "[1] 2.2.4. TOE Interfaces | p.15 | Score: 0.87"
      "[RESUME] 1. INTRODUCTION (9 chunks) | p.3"
      "[3] Section 5 / 0 | p.12 | Score: 0.65"
    """
    heading = meta.get("heading", "")
    breadcrumb = meta.get("breadcrumb", "")
    chunk_type = meta.get("chunk_type", "chunk")
    page_num = meta.get("page_number")
    source = meta.get("source", "")
    section_idx = meta.get("section_idx", "?")
    chunk_idx = meta.get("chunk_idx", "?")
    summary_n = meta.get("summary_num_chunks")

    parts = []

    # Rang (tab2)
    if rank is not None:
        parts.append(f"[{rank}]")

    # Type badge
    if chunk_type == "summary":
        n_str = f" ({summary_n} chunks)" if summary_n else ""
        title = heading or breadcrumb or source
        parts.append(f"[RESUME] {title}{n_str}")
    elif chunk_type in ("table", "figure", "mixed"):
        type_badge = {"table": "[TAB]", "figure": "[FIG]", "mixed": "[TAB+FIG]"}.get(chunk_type, "")
        title = heading or breadcrumb or source or f"Section {section_idx}"
        parts.append(f"{type_badge} {title}")
    elif heading:
        parts.append(heading)
    elif breadcrumb:
        parts.append(breadcrumb)
    else:
        # Fallback : source + identifiant technique
        label = source if source else f"Section {section_idx}"
        if chunk_idx not in ("?", 0, "0"):
            label += f" / {chunk_idx}"
        parts.append(label)

    # Page
    if page_num is not None:
        parts.append(f"p.{page_num}")

    # Score (tab2)
    if score is not None:
        parts.append(f"Score: {score:.2f}")

    return " | ".join(parts)


def _chunk_detail_block(meta: dict, content: str, show_content: bool = True):
    """
    Affiche le contenu et les métadonnées enrichies d'un chunk dans un expander.
    Utilisé dans tab2 et tab3 pour un affichage cohérent.
    """
    chunk_type = meta.get("chunk_type", "chunk")
    source = meta.get("source", "")
    heading = meta.get("heading", "")
    breadcrumb = meta.get("breadcrumb", "")
    page_num = meta.get("page_number")
    kw = meta.get("keywords_str", "") or meta.get("keywords", "")
    qq = meta.get("questions_str", "") or meta.get("questions", "")
    ent_str = meta.get("entities_str", "")
    # Convertir les listes en strings si nécessaire
    if isinstance(kw, list):
        kw = ", ".join(kw)
    if isinstance(qq, list):
        qq = " | ".join(qq)
    summary_n = meta.get("summary_num_chunks")

    # Barre de contexte (toujours visible, avant le contenu)
    info_parts = []
    if source:
        info_parts.append(source)
    if page_num is not None:
        info_parts.append(f"p.{page_num}")
    if breadcrumb:
        info_parts.append(breadcrumb)
    if info_parts:
        st.caption(" · ".join(info_parts))

    # Contenu principal
    if show_content:
        st.markdown(content)

    # Métadonnées enrichies
    has_extra = bool(kw or qq or ent_str or (chunk_type == "summary") or chunk_type in ("table", "figure", "mixed"))
    if has_extra:
        st.markdown("---")
        if chunk_type == "summary":
            st.markdown(f"**Résumé RAPTOR** de {summary_n or '?'} chunks")
        if chunk_type in ("table", "figure", "mixed"):
            type_label = {"table": "Tableau", "figure": "Figure", "mixed": "Mixte (tableau + figure)"}.get(chunk_type, chunk_type)
            st.markdown(f"**Type :** {type_label}")
            table_desc = meta.get("table_description", "")
            if table_desc:
                st.markdown(f"**Description :** {table_desc}")
        if kw:
            st.markdown(f"**Mots-clés :** {kw}")
        if qq:
            st.markdown(f"**Questions :** {qq}")
        if ent_str:
            st.markdown(f"**Entites :** {ent_str}")


def _display_citations(citations: list[dict]):
    """
    Affichage unifié du panneau de citations / sources.
    Utilisé dans tab1 (réponse brute) et tab2 (réponse sélection).
    """
    if not citations:
        return
    st.write("### Sources citées")
    for cit in citations:
        parts = [f"**[{cit['idx']}]**"]
        # Heading ou source
        heading = cit.get("heading", "")
        source = cit.get("source", "")
        chunk_type = cit.get("chunk_type", "chunk")
        if chunk_type == "summary":
            parts.append(f"[RESUME] {heading or source}")
        elif heading:
            parts.append(heading)
        else:
            parts.append(source)
        # Page
        if cit.get("page"):
            parts.append(f"p.{cit['page']}")
        # Breadcrumb
        bc = cit.get("breadcrumb", "")
        if bc:
            parts.append(f"_{bc}_")
        st.markdown("&ensp;" + " · ".join(parts))

# Configuration de la page Streamlit
st.set_page_config(
    page_title="SSH GPT - RAG System",
    page_icon="/home/marsattacks/Downloads/logoSSH-GPT.png",
    layout="wide"
)

# ── Design épuré & professionnel ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Variables ── */
    :root {
        --bg-app:        #1e1e2e;
        --bg-sidebar:    #16162a;
        --bg-chat:       #23233a;
        --bg-card:       #2a2a42;
        --accent:        #4f8ef7;
        --accent-hover:  #3a73d6;
        --accent-dim:    rgba(79,142,247,0.15);
        --accent-border: rgba(79,142,247,0.38);
        --text-primary:  #e8e8f0;
        --text-muted:    #9090b0;
        --divider:       rgba(79,142,247,0.20);
        --radius:        8px;
        --font:          'Inter', 'Segoe UI', system-ui, sans-serif;
    }

    /* ── Fond + couleur de base héritée ── */
    .stApp {
        background-color: var(--bg-app) !important;
        font-family: var(--font) !important;
        color: var(--text-primary) !important;
        color-scheme: dark;
    }
    section[data-testid="stMain"] { background-color: var(--bg-chat) !important; }

    /* ── Propagation du texte clair à tous les descendants directs ── */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stVerticalBlock"],
    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stHorizontalBlock"],
    [data-testid="stColumn"],
    .main .block-container {
        color: var(--text-primary) !important;
    }

    /* ── Texte Streamlit générique (st.write, st.text, st.markdown…) ── */
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    [data-testid="stText"],
    [data-testid="stWrite"] {
        color: var(--text-primary) !important;
        font-family: var(--font) !important;
        text-align: left !important;
    }
    .stMarkdown p { font-size: 14px !important; line-height: 1.6 !important; }

    /* ── Titres ── */
    h1 { font-size: 26px !important; font-weight: 700 !important; color: var(--text-primary) !important; text-align: left !important; margin: 0.4rem 0 0.8rem !important; }
    h2 { font-size: 20px !important; font-weight: 600 !important; color: var(--text-primary) !important; text-align: left !important; }
    h3, h4 { font-size: 16px !important; font-weight: 600 !important; color: var(--text-primary) !important; text-align: left !important; }

    /* ── Labels widgets ── */
    .stTextInput label, .stTextArea label,
    .stSelectbox label, .stFileUploader label,
    .stNumberInput label, .stSlider label,
    .stRadio label, .stMultiSelect label {
        color: var(--text-muted) !important;
        font-size: 12px !important;
        font-family: var(--font) !important;
    }
    .stCheckbox label { color: var(--text-primary) !important; font-size: 13px !important; }
    /* Forçage spécifique pour les checkboxes dans tous les contextes (sidebar, colonnes sombres) */
    .stCheckbox label p,
    .stCheckbox span[data-testid="stWidgetLabel"],
    .stCheckbox [data-testid="stWidgetLabel"] p,
    .stCheckbox [data-testid="stWidgetLabel"] span,
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] label p,
    [data-testid="stCheckbox"] p {
        color: var(--text-primary) !important;
        font-size: 13px !important;
        font-family: var(--font) !important;
    }

    /* ── Contenu sélectionné des selectbox (valeur affichée) ── */
    .stSelectbox [data-baseweb="select"] [data-testid="stWidgetLabel"] ~ div,
    .stSelectbox [data-baseweb="select"] > div > div {
        color: var(--text-primary) !important;
        background-color: var(--bg-card) !important;
    }
    /* Valeur texte dans la selectbox fermée */
    .stSelectbox span[data-baseweb="tag"],
    .stSelectbox div[data-baseweb="select"] span {
        color: var(--text-primary) !important;
    }

    /* ── Onglets ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--bg-sidebar) !important;
        border-radius: var(--radius);
        padding: 4px 6px;
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-muted) !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        background-color: transparent !important;
        border-radius: 6px !important;
        padding: 7px 16px !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: var(--accent-dim) !important; color: var(--text-primary) !important; }
    .stTabs [aria-selected="true"] { background-color: var(--accent) !important; color: #fff !important; font-weight: 600 !important; }

    /* ── Boutons ── */
    .stButton > button {
        background-color: var(--accent) !important;
        color: #fff !important;
        border: none !important;
        border-radius: var(--radius) !important;
        padding: 8px 16px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        width: 100% !important;
    }
    .stButton > button:hover { background-color: var(--accent-hover) !important; }
    .stButton > button[kind="secondary"] {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--accent-border) !important;
    }
    .stButton > button[kind="secondary"]:hover { background-color: var(--accent-dim) !important; }

    /* ── Inputs texte ── */
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        background-color: rgba(245,245,250,0.95) !important;
        color: #1e1e2e !important;
        border: 1.5px solid var(--accent-border) !important;
        border-radius: var(--radius) !important;
        font-size: 14px !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(79,142,247,0.15) !important;
    }

    /* ── Chat input — style Apple ── */
    [data-testid="stChatInput"] {
        background: transparent !important;
    }
    [data-testid="stChatInput"] > div {
        background-color: #ffffff !important;
        border-radius: 18px !important;
        box-shadow: 0 2px 16px rgba(0,0,0,0.18), 0 1px 4px rgba(0,0,0,0.10) !important;
        border: none !important;
        padding: 2px 6px !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
        border: none !important;
        border-radius: 16px !important;
        font-size: 15px !important;
        font-family: var(--font) !important;
        caret-color: var(--accent) !important;
        padding: 12px 16px !important;
        line-height: 1.5 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #aaaabc !important;
        font-size: 14px !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    /* Bouton d'envoi */
    [data-testid="stChatInput"] button {
        background-color: var(--accent) !important;
        border-radius: 50% !important;
        border: none !important;
        width: 34px !important;
        height: 34px !important;
        min-width: 34px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 8px rgba(79,142,247,0.35) !important;
    }
    [data-testid="stChatInput"] button:hover {
        background-color: var(--accent-hover) !important;
    }
    [data-testid="stChatInput"] button svg { stroke: #fff !important; }

    /* ── Messages chat — style Apple ── */
    [data-testid="stChatMessage"] {
        border: none !important;
        background: transparent !important;
        padding: 6px 0 !important;
    }
    [data-testid="stChatMessage"][data-role="user"] {
        background: transparent !important;
    }
    [data-testid="stChatMessage"][data-role="user"] > div:last-child {
        background-color: #ffffff !important;
        border-radius: 18px 18px 6px 18px !important;
        padding: 12px 18px !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.13) !important;
        max-width: 80% !important;
        margin-left: auto !important;
        color: #1a1a2e !important;
    }
    [data-testid="stChatMessage"][data-role="assistant"] {
        background: transparent !important;
        border-left: none !important;
    }
    [data-testid="stChatMessage"][data-role="assistant"] > div:last-child {
        background-color: var(--bg-card) !important;
        border-radius: 18px 18px 18px 6px !important;
        padding: 14px 18px !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.18) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        max-width: 90% !important;
    }
    /* Texte dans les bulles de chat */
    [data-testid="stChatMessage"][data-role="user"] p,
    [data-testid="stChatMessage"][data-role="user"] span {
        color: #1a1a2e !important;
        font-size: 14px !important;
        line-height: 1.55 !important;
    }
    [data-testid="stChatMessage"][data-role="assistant"] p,
    [data-testid="stChatMessage"][data-role="assistant"] span,
    [data-testid="stChatMessage"][data-role="assistant"] li {
        color: var(--text-primary) !important;
        font-size: 14px !important;
        line-height: 1.65 !important;
    }

    /* ── Expanders ── */
    .stExpander {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--divider) !important;
        border-radius: var(--radius) !important;
    }
    /* Titre de l'expander */
    .stExpander summary p,
    .stExpander [data-testid="stExpanderToggleIcon"] + span,
    details summary span {
        color: var(--text-primary) !important;
        font-size: 13px !important;
    }
    /* Contenu à l'intérieur des expanders */
    .stExpander [data-testid="stVerticalBlock"] p,
    .stExpander [data-testid="stVerticalBlock"] span,
    .stExpander [data-testid="stVerticalBlock"] li,
    .stExpander .stMarkdown p {
        color: var(--text-primary) !important;
    }

    /* ── Captions ── */
    .stCaption, [data-testid="stCaptionContainer"] p { color: var(--text-muted) !important; font-size: 11px !important; }

    /* ── Métriques ── */
    [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-size: 24px !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 11px !important; }

    /* ── Alertes ── */
    .stSuccess, div[data-testid="stNotificationContentSuccess"] { background-color: rgba(76,175,80,0.12) !important; border-left: 3px solid #4caf50 !important; border-radius: var(--radius) !important; }
    .stError,   div[data-testid="stNotificationContentError"]   { background-color: rgba(239,83,80,0.12) !important;  border-left: 3px solid #ef5350 !important;  border-radius: var(--radius) !important; }
    .stWarning, div[data-testid="stNotificationContentWarning"] { background-color: rgba(255,167,38,0.12) !important; border-left: 3px solid #ffa726 !important; border-radius: var(--radius) !important; }
    .stInfo,    div[data-testid="stNotificationContentInfo"]    { background-color: rgba(66,165,245,0.12) !important; border-left: 3px solid #42a5f5 !important; border-radius: var(--radius) !important; }
    /* Texte dans les alertes */
    .stSuccess p, .stError p, .stWarning p, .stInfo p,
    [data-testid*="stNotificationContent"] p,
    [data-testid*="stNotificationContent"] span { color: var(--text-primary) !important; }

    /* ── Dividers ── */
    hr { border: none !important; border-top: 1px solid var(--divider) !important; margin: 0.7rem 0 !important; }

    /* ── Progress ── */
    .stProgress > div { background-color: var(--bg-card) !important; border-radius: 4px !important; }
    .stProgress > div > div { background-color: var(--accent) !important; border-radius: 4px !important; }

    /* ── Scrollbar fine ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(79,142,247,0.25); border-radius: 10px; }

    /* ── Tooltips ── */
    div[data-testid="stTooltipContent"] {
        background-color: #2a2a42 !important;
        border: 1px solid var(--accent-border) !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.45) !important;
    }
    div[data-testid="stTooltipContent"] p,
    div[data-testid="stTooltipContent"] span {
        color: #e8e8f0 !important;
        font-size: 13px !important;
        line-height: 1.55 !important;
    }

    /* ── Bouton flottant "Historique" ── */
    .sessions-toggle-btn {
        position: fixed !important;
        top: 72px;
        left: 16px;
        z-index: 1200;
    }
    .sessions-toggle-btn button {
        background: var(--bg-card) !important;
        border: 1px solid var(--accent-border) !important;
        border-radius: 20px !important;
        color: var(--text-primary) !important;
        font-size: 12px !important;
        padding: 5px 14px !important;
        cursor: pointer;
        box-shadow: 0 2px 10px rgba(0,0,0,0.35) !important;
        transition: background 0.15s, box-shadow 0.15s;
    }
    .sessions-toggle-btn button:hover {
        background: var(--accent-dim) !important;
        box-shadow: 0 2px 16px rgba(79,142,247,0.25) !important;
    }

    /* ── Overlay sombre derrière le panneau ── */
    #sessions-overlay {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.45);
        z-index: 1300;
        backdrop-filter: blur(2px);
        -webkit-backdrop-filter: blur(2px);
    }

    /* ── Panneau flottant sessions ── */
    #sessions-panel {
        position: fixed;
        top: 0;
        left: 0;
        height: 100vh;
        width: 300px;
        background: var(--bg-sidebar);
        border-right: 2px solid var(--accent-border);
        box-shadow: 4px 0 28px rgba(0,0,0,0.55);
        z-index: 1400;
        overflow-y: auto;
        padding: 20px 16px 32px;
        transform: translateX(-110%);
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        scrollbar-width: thin;
        scrollbar-color: rgba(79,142,247,0.25) transparent;
    }
    #sessions-panel.open {
        transform: translateX(0);
    }

    /* Bouton fermer (✕) dans le coin du panneau */
    #sessions-close-btn {
        position: absolute;
        top: 14px;
        right: 14px;
        background: transparent;
        border: none;
        color: var(--text-muted);
        font-size: 18px;
        cursor: pointer;
        line-height: 1;
        padding: 2px 6px;
        border-radius: 4px;
        transition: color 0.15s, background 0.15s;
    }
    #sessions-close-btn:hover {
        color: var(--text-primary);
        background: var(--accent-dim);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo en haut de la page
col1, col2, col3 = st.columns([2, 2, 1])
with col2:
    st.image("/home/marsattacks/Downloads/logoSSH-GPT.png", width=300)

st.markdown("<hr>", unsafe_allow_html=True)

def get_first_key(chunk, keys, default="N/A"):
    for k in keys:
        if k in chunk:
            return chunk[k]
    return default

@st.cache_resource
def get_mongo_collection():
    client = MongoClient("mongodb://localhost:27017")
    db = client["ragdb"]
    return db["chunks"]

col = get_mongo_collection()

# Initialiser la session
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
if "last_ingested_md" not in st.session_state:
    st.session_state.last_ingested_md = None
if "last_ingested_source" not in st.session_state:
    st.session_state.last_ingested_source = None
# Pour stocker les chunks à afficher dans le formulaire de sélection
if "final_chunks" not in st.session_state:
    st.session_state.final_chunks = None
if "citations" not in st.session_state:
    st.session_state.citations = []
if "last_query" not in st.session_state:
    st.session_state.last_query = None

# On initialise l'historique des requêtes
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# Sessions de chat persistées (style ChatGPT)
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None   # None = aucune session ouverte
if "use_memory" not in st.session_state:
    st.session_state.use_memory = False         # mémoire conversationnelle désactivée par défaut
if "show_sidebar" not in st.session_state:
    st.session_state.show_sidebar = True        # panneau sessions visible par défaut

# Self-RAG
from config import SELF_RAG_ENABLED, SELF_RAG_THRESHOLD, SELF_RAG_MAX_RETRIES, PARENT_CHILD_ENABLED
from config import CE_RELEVANCE_THRESHOLD
if "self_rag_enabled" not in st.session_state:
    st.session_state.self_rag_enabled     = False  # désactivé par défaut (latence)
if "self_rag_threshold" not in st.session_state:
    st.session_state.self_rag_threshold   = SELF_RAG_THRESHOLD
if "self_rag_max_retries" not in st.session_state:
    st.session_state.self_rag_max_retries = SELF_RAG_MAX_RETRIES
if "ce_relevance_threshold" not in st.session_state:
    st.session_state.ce_relevance_threshold = CE_RELEVANCE_THRESHOLD

# Parent-Child Retrieval
if "parent_child_enabled" not in st.session_state:
    st.session_state.parent_child_enabled = False  # désactivé par défaut (latence)

# Réécriture LLM de la requête
if "rewrite_enabled" not in st.session_state:
    st.session_state.rewrite_enabled = False  # désactivé par défaut (latence ~3-5s)

# Modèles actifs (Settings)
from config import NUM_CHUNKS, EMBED_MODEL, GEN_MODEL
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = NUM_CHUNKS
if "embed_model" not in st.session_state:
    st.session_state.embed_model = EMBED_MODEL
if "gen_model" not in st.session_state:
    st.session_state.gen_model = GEN_MODEL

# Flag génération en cours (empêche double-requête sur rerun checkbox)
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

# Paramètres d'enrichissement
if "auto_keywords" not in st.session_state:
    st.session_state.auto_keywords = AUTO_KEYWORDS
if "auto_questions" not in st.session_state:
    st.session_state.auto_questions = AUTO_QUESTIONS
if "chunking_mode" not in st.session_state:
    st.session_state.chunking_mode = CHUNKING_MODE
if "raptor_summaries" not in st.session_state:
    st.session_state.raptor_summaries = RAPTOR_SUMMARIES

# Etat ingestion en arriere-plan
if "ingest_running" not in st.session_state:
    st.session_state.ingest_running = False
if "ingest_progress" not in st.session_state:
    st.session_state.ingest_progress = 0
if "ingest_step" not in st.session_state:
    st.session_state.ingest_step = ""
if "ingest_result" not in st.session_state:
    st.session_state.ingest_result = None  # dict stats ou {"error": ...}
if "ingest_thread" not in st.session_state:
    st.session_state.ingest_thread = None

# Auto-refresh global si ingestion en cours (toutes les 1s pour un suivi reactif)
if st.session_state.ingest_running:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1000, limit=None, key="ingest_autorefresh")
    _pct = st.session_state.ingest_progress
    _step = st.session_state.ingest_step
    # Parser current/total pour le bandeau
    import re as _re_banner
    _bm = _re_banner.search(r'(\d+)\s*/\s*(\d+)', _step)
    _b_detail = f" ({_bm.group(1)}/{_bm.group(2)})" if _bm else ""
    # Bandeau fixe en haut de page, visible depuis n'importe quel onglet
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, #1565c0 0%, #0d47a1 100%);
        color: #e3f2fd;
        padding: 0.45rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        font-size: 0.93rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        border: 1px solid #42a5f5;
    ">
        <span style="font-weight:600;color:#90caf9;">Ingestion en cours</span>
        <span style="
            background: #42a5f5; color: #fff; font-weight:700;
            padding: 0.1rem 0.55rem; border-radius: 10px; font-size: 0.88rem;
        ">{_pct}%{_b_detail}</span>
        <span style="color:#bbdefb;">{_step}</span>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["SSH GPT", "Exploration chunks", "Pipeline RAG", "Systeme Prompt", "Settings", "Graphe d'entites"])
with tab1:
    # ── Initialisation session state ───────────────────────────────────────────
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "chat_sources_filter" not in st.session_state:
        st.session_state.chat_sources_filter = None
    if "eval_result" not in st.session_state:
        st.session_state.eval_result = None   # résultat de la dernière évaluation

    # ── Layout 2 colonnes : sidebar sessions | zone de chat ──────────────────
    col_sessions, col_chat = st.columns([1, 3], gap="small")

    with col_sessions:
        st.markdown(
            "<p style='font-size:11px;font-weight:600;text-transform:uppercase;"
            "letter-spacing:0.8px;color:#9090b0;margin:0 0 10px;'>Conversations</p>",
            unsafe_allow_html=True
        )

        if st.button("+ Nouvelle conversation", use_container_width=True, key="btn_new_session"):
            new_sid = create_session(source_filter=st.session_state.chat_sources_filter)
            st.session_state.active_session_id = new_sid
            st.session_state.eval_result = None
            st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:11px;font-weight:600;text-transform:uppercase;"
            "letter-spacing:0.8px;color:#9090b0;margin:0 0 8px;'>Options pipeline</p>",
            unsafe_allow_html=True
        )

        st.checkbox(
            "Mémoire conversationnelle",
            key="use_memory",
            disabled=st.session_state.is_generating,
            help=(
                "Active : le LLM tient compte des echanges precedents et reformule "
                "les questions de suivi en questions autonomes.\n\n"
                "Desactive : chaque question est traitee independamment."
            ),
        )

        st.checkbox(
            "Self-RAG",
            key="self_rag_enabled",
            disabled=st.session_state.is_generating,
            help=(
                "Active : apres generation, le LLM evalue automatiquement sa reponse "
                "(Fidélité, Pertinence réponse, Pertinence contexte).\n\n"
                "Si le score est insuffisant, le pipeline reformule la question et retente "
                "le retrieval+generation (max 1 retry par defaut).\n\n"
                "Desactive : pipeline classique, pas d'evaluation automatique.\n\n"
                "Attention : ajoute ~10-20s de latence par appel (3 appels LLM supplementaires)."
            ),
        )

        st.checkbox(
            "Parent-Child (Small-to-Big)",
            key="parent_child_enabled",
            disabled=st.session_state.is_generating,
            help=(
                "Active : apres le reranking, chaque chunk recupere est elargi "
                "au contexte complet de sa section parente (jusqu'a 8000 chars).\n\n"
                "Le LLM recoit ainsi un contexte plus riche sans degrader la precision "
                "du retrieval (les scores restent ceux des chunks enfants).\n\n"
                "Desactive : le LLM recoit uniquement le chunk exact recupere.\n\n"
                "Impact latence : negligeable (~3-5ms, 1 requete MongoDB groupee)."
            ),
        )

        st.checkbox(
            "Réécriture requête (LLM)",
            key="rewrite_enabled",
            disabled=st.session_state.is_generating,
            help=(
                "Active : la requête est réécrite et nettoyée par un LLM (llama3.1) "
                "avant le retrieval. Améliore la qualité sur les questions vagues "
                "ou avec fautes.\n\n"
                "Désactivé : la requête est envoyée directement au retrieval "
                "(nettoyage léger sans LLM).\n\n"
                "Impact latence : +3 à 5 secondes quand activé."
            ),
        )

        # ── Modèle LLM en VRAM ─────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:11px;font-weight:600;text-transform:uppercase;"
            "letter-spacing:0.8px;color:#9090b0;margin:0 0 8px;'>Modèle en VRAM</p>",
            unsafe_allow_html=True,
        )
        try:
            import requests as _rq
            _ps = _rq.get("http://localhost:11434/api/ps", timeout=2).json()
            _in_vram = [m["name"] for m in _ps.get("models", [])]
        except Exception:
            _in_vram = []

        _cur_model = st.session_state.gen_model
        _is_loaded = _cur_model in _in_vram

        if _is_loaded:
            st.markdown(
                f"<span style='color:#4ade80;font-size:12px'>🟢 {_cur_model}</span>"
                f"<span style='color:#9090b0;font-size:11px'> · en VRAM</span>",
                unsafe_allow_html=True,
            )
            if st.button("⏏ Décharger", key="sb_unload_llm",
                         use_container_width=True,
                         disabled=st.session_state.is_generating):
                try:
                    _rq.post("http://localhost:11434/api/generate",
                             json={"model": _cur_model, "keep_alive": 0, "prompt": ""},
                             timeout=10)
                    st.rerun()
                except Exception as _ex:
                    st.error(str(_ex))
        else:
            st.markdown(
                f"<span style='color:#9090b0;font-size:12px'>⚫ {_cur_model}</span>"
                f"<span style='color:#9090b0;font-size:11px'> · non chargé</span>",
                unsafe_allow_html=True,
            )
            if st.button("▶ Charger en VRAM", key="sb_load_llm",
                         use_container_width=True,
                         disabled=st.session_state.is_generating,
                         help="Précharge le modèle pour éviter le cold-start (~10s)"):
                with st.spinner(f"Chargement de {_cur_model}…"):
                    try:
                        import time as _tm
                        _t0 = _tm.time()
                        _rq.post("http://localhost:11434/api/generate",
                                 json={"model": _cur_model, "prompt": "",
                                       "keep_alive": -1,
                                       "options": {"num_predict": 0}},
                                 timeout=120)
                        _elapsed = _tm.time() - _t0
                        st.success(f"✅ Chargé en {_elapsed:.1f}s")
                        st.rerun()
                    except Exception as _ex:
                        st.error(str(_ex))

        st.markdown("---")

        all_sessions = list_sessions(limit=30)
        if not all_sessions:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.caption("Aucune session pour le moment.")
        else:
            st.markdown("<hr>", unsafe_allow_html=True)
            for sess in all_sessions:
                sid   = sess["session_id"]
                title = sess.get("title", "Conversation")
                is_active = (sid == st.session_state.active_session_id)

                scol1, scol2 = st.columns([5, 1])
                with scol1:
                    btn_label = f"● {title}" if is_active else title
                    if st.button(btn_label, key=f"sess_{sid}", use_container_width=True,
                                 type="primary" if is_active else "secondary"):
                        st.session_state.active_session_id = sid
                        st.session_state.eval_result = None
                        st.rerun()
                with scol2:
                    if st.button("✕", key=f"del_{sid}"):
                        delete_session(sid)
                        if st.session_state.active_session_id == sid:
                            st.session_state.active_session_id = None
                        st.rerun()


    # ── Zone de chat ──────────────────────────────────────────────────────────
    with col_chat:
        # En-tête épuré
        st.markdown(
            "<h1 style='margin-bottom:4px;'>SSH GPT</h1>"
            "<p style='color:#9090b0;font-size:13px;margin-top:0;margin-bottom:16px;'>"
            "Posez vos questions sur les documents ingérés.</p>",
            unsafe_allow_html=True
        )

        # Sélecteur de document
        sources_chat = col.distinct("source")
        if sources_chat:
            selected_source_label_chat = st.selectbox(
                "Document :",
                ["Tous les documents"] + sources_chat,
                key="chat_source_select",
                help="Restreint la recherche au document selectionne.",
            )
            st.session_state.chat_sources_filter = (
                None if selected_source_label_chat == "Tous les documents"
                else selected_source_label_chat
            )
            with st.expander("Documents disponibles", expanded=False):
                for src in sources_chat:
                    n_chunks = col.count_documents({"source": src})
                    st.write(f"**{src}** — {n_chunks} chunks")
        else:
            st.session_state.chat_sources_filter = None
            st.info("Aucun document ingere. Allez dans l'onglet Pipeline RAG.")

        # Session active : créer si besoin
        if st.session_state.active_session_id is None:
            recent = list_sessions(limit=1)
            if recent:
                st.session_state.active_session_id = recent[0]["session_id"]
            else:
                new_sid = create_session(source_filter=st.session_state.chat_sources_filter)
                st.session_state.active_session_id = new_sid

        active_sid = st.session_state.active_session_id

        # Sync filtre source
        sess_data = get_session(active_sid) or {}
        if sess_data.get("source_filter") != st.session_state.chat_sources_filter:
            update_session_source(active_sid, st.session_state.chat_sources_filter)

        messages = get_messages(active_sid)

        # Bouton effacer
        if messages:
            if st.button("Effacer la conversation", key="clear_active_chat"):
                clear_session_messages(active_sid)
                st.session_state.eval_result = None
                st.rerun()

        st.markdown("---")

        # ── Affichage des messages ────────────────────────────────────────────
        for i, msg in enumerate(messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Citations uniquement sur les messages de l'historique (pas le dernier)
                if msg["role"] == "assistant" and msg.get("citations") and i < len(messages) - 1:
                    with st.expander("Sources", expanded=False):
                        _display_citations(msg["citations"])

        # ── Dernière réponse : Sources + Chunks interactifs ───────────────────
        last_assistant = next(
            (m for m in reversed(messages) if m["role"] == "assistant"), None
        )
        last_user = next(
            (m for m in reversed(messages) if m["role"] == "user"), None
        )

        if last_assistant and st.session_state.final_chunks:
            with st.expander(
                f"Sources & Chunks ({len(st.session_state.final_chunks)} recuperes) — cocher/decocher puis regenerer",
                expanded=False,
            ):
                with st.form("chunk_regen_form"):
                    chunk_checks = []
                    for i, chunk in enumerate(st.session_state.final_chunks):
                        meta    = chunk.get("meta", {})
                        score   = chunk.get("score_global", chunk.get("score", 0))
                        content = (
                            chunk.get("content")
                            or chunk.get("doc")
                            or chunk.get("page_content")
                            or chunk.get("text")
                            or "Contenu non disponible"
                        )
                        header = _chunk_display_label(meta, rank=i + 1, score=score)
                        checked = st.checkbox(header, value=True, key=f"chunk_{i}_{st.session_state.get('checkbox_counter', 0)}")
                        if checked:
                            chunk_checks.append(chunk)
                        with st.expander(f"Details — {header}", expanded=False):
                            _chunk_detail_block(meta, content)

                    regen_submitted = st.form_submit_button("Regenerer la reponse avec les chunks selectionnes")

                if regen_submitted:
                    if not chunk_checks:
                        st.warning("Selectionnez au moins un chunk.")
                    else:
                        regen_query = last_user["content"] if last_user else st.session_state.last_query or ""
                        with st.spinner("Generation en cours..."):
                            if st.session_state.self_rag_enabled:
                                from core.self_rag import self_rag_query
                                regen_rep, regen_chunks, regen_cit, _ = self_rag_query(
                                    regen_query,
                                    system_prompt=st.session_state.system_prompt,
                                    source_filter=st.session_state.chat_sources_filter,
                                    conversation_history=(
                                        [{"role": m["role"], "content": m["content"]} for m in messages]
                                        if st.session_state.use_memory else []
                                    ),
                                )
                            else:
                                from core.ask import process_query as _pq
                                regen_rep, regen_chunks, regen_cit = _pq(
                                    regen_query,
                                    selected_chunks=chunk_checks,
                                    system_prompt=st.session_state.system_prompt,
                                )
                        # Mise à jour session state + sauvegarde MongoDB
                        st.session_state.final_chunks = regen_chunks or chunk_checks
                        st.session_state.citations    = regen_cit or []
                        add_message(active_sid, "assistant", regen_rep, citations=regen_cit)
                        st.session_state.eval_result = None
                        st.rerun()

        if last_assistant and last_assistant.get("content") and st.session_state.final_chunks:
            st.markdown("---")
            run_eval = st.button("Evaluate this answer", key="btn_eval_last", type="secondary")

            if run_eval:
                with st.spinner("Running LLM-as-judge evaluation..."):
                    from core.evaluation import evaluate_single
                    eval_metrics = evaluate_single(
                        question=last_user["content"] if last_user else "",
                        generated_answer=last_assistant["content"],
                        retrieved_chunks=st.session_state.final_chunks,
                        reference_answer="",
                        use_llm_judge=True,
                    )
                st.session_state.eval_result = eval_metrics

        # Affichage du résultat d'évaluation
        if st.session_state.eval_result:
            ev = st.session_state.eval_result
            st.markdown("#### Évaluation automatique *(LLM-as-judge, score 0 → 1)*")

            METRIC_INFO = {
                "Fidélité": (
                    ev.get("faithfulness"),
                    "La réponse ne contient-elle que des faits issus du contexte récupéré ? "
                    "Score faible = risque d'hallucination.",
                ),
                "Pertinence réponse": (
                    ev.get("answer_relevance"),
                    "La réponse répond-elle bien à la question posée ? "
                    "Score faible = réponse hors sujet ou incomplète.",
                ),
                "Pertinence contexte": (
                    ev.get("context_relevance"),
                    "Les passages récupérés sont-ils utiles pour répondre à la question ? "
                    "Score faible = le retrieval remonte des passages non pertinents.",
                ),
            }

            available = [(name, val, desc) for name, (val, desc) in METRIC_INFO.items() if val is not None]
            if available:
                m_cols = st.columns(len(available))
                for col_idx, (name, val, desc) in enumerate(available):
                    with m_cols[col_idx]:
                        st.metric(name, f"{val:.2f}")
                        st.caption(desc)

            st.caption(f"Passages évalués : {ev.get('num_chunks_retrieved', 0)}")

            if st.button("Fermer l'évaluation", key="btn_close_eval"):
                st.session_state.eval_result = None
                st.rerun()

        # ── Input chatbot ─────────────────────────────────────────────────────
        # Bloquer visuellement le chat_input pendant une génération en cours
        if st.session_state.is_generating:
            st.chat_input("Génération en cours...", key="chat_input_disabled", disabled=True)
            user_input = None
        else:
            user_input = st.chat_input("Posez votre question...", key="chat_input_main")

        if user_input:
            selected_source_filter = st.session_state.chat_sources_filter
            use_memory = st.session_state.use_memory
            history_for_llm = (
                [{"role": m["role"], "content": m["content"]} for m in messages]
                if use_memory else []
            )

            # Snapshot IMMÉDIAT de toutes les options pipeline au moment de l'envoi.
            # Ces valeurs sont figées pour toute la durée du traitement.
            _snapshot = {
                "parent_child_on":  st.session_state.get("parent_child_enabled", False),
                "rewrite_enabled":  st.session_state.get("rewrite_enabled", False),
                "system_prompt":    st.session_state.system_prompt,
                "source_filter":    selected_source_filter,
            }

            with st.chat_message("user"):
                st.markdown(user_input)

            save_query_to_mongo(user_input)
            if user_input not in st.session_state.query_history:
                st.session_state.query_history.insert(0, user_input)
                if len(st.session_state.query_history) > 10:
                    st.session_state.query_history.pop()

            if "checkbox_counter" not in st.session_state:
                st.session_state.checkbox_counter = 0
            st.session_state.checkbox_counter += 1

            # Lever le verrou — désactive chat_input + checkboxes dès ce rerun
            st.session_state.is_generating = True

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                status_placeholder   = st.empty()

                try:
                    status_placeholder.caption("Retrieval en cours...")
                    token_gen, final_chunks, citations = process_query_stream(
                        user_input,
                        system_prompt=_snapshot["system_prompt"],
                        source_filter=_snapshot["source_filter"],
                        conversation_history=history_for_llm,
                        parent_child_on=_snapshot["parent_child_on"],
                        rewrite_enabled=_snapshot["rewrite_enabled"],
                    )
                    st.session_state.final_chunks = final_chunks or []
                    st.session_state.citations    = citations or []
                    st.session_state.last_query   = user_input

                    if token_gen is None:
                        status_placeholder.empty()
                        response_placeholder.warning(
                            "Aucun chunk pertinent trouve. "
                            "Essayez une autre formulation ou selectionnez un autre document."
                        )
                        full_response = ""
                    else:
                        status_placeholder.caption("Génération en cours...")
                        full_response = ""
                        _is_out_of_scope = False
                        try:
                            for token in token_gen:
                                full_response += token
                                response_placeholder.markdown(full_response + "▌")
                        except (BrokenPipeError, StopIteration, GeneratorExit):
                            pass
                        except Exception as stream_err:
                            print(f"[stream] Interruption : {stream_err}")
                        response_placeholder.markdown(full_response)
                        status_placeholder.empty()

                        # Détection hors-scope : message commence par l'emoji d'avertissement
                        if full_response.startswith("⚠️"):
                            _is_out_of_scope = True
                            st.session_state.final_chunks = []
                            st.session_state.citations    = []

                        if citations and not _is_out_of_scope:
                            with st.expander("Sources", expanded=False):
                                _display_citations(citations)

                except Exception as e:
                    status_placeholder.empty()
                    response_placeholder.error(f"Erreur : {e}")
                    full_response = f"[Erreur : {e}]"
                    citations = []
                finally:
                    # Toujours libérer le verrou en fin de génération
                    st.session_state.is_generating = False

            add_message(active_sid, "user", user_input)
            add_message(active_sid, "assistant", full_response, citations=citations)
            st.session_state.eval_result = None

            st.session_state.conversation_history = [
                {"role": m["role"], "content": m["content"]}
                for m in get_messages(active_sid)
            ]
            st.rerun()





with tab2:
    st.title("Explorer les chunks")
    import html as _html_mod
    import markdown2 as _md2

    # ── Helpers internes ──
    CLEAN_MD_ROOT = Path(__file__).resolve().parent.parent / "data" / "out_clean_md"

    def _load_md_text(source_name: str) -> str:
        md_path = CLEAN_MD_ROOT / source_name
        if md_path.exists():
            return md_path.read_text(encoding="utf-8")
        return ""

    def _find_chunk_in_md(md_text: str, chunk_content: str) -> tuple[int, int]:
        if not chunk_content or not md_text:
            return -1, -1
        clean = chunk_content.strip()
        if clean.startswith("[") and "]" in clean:
            bracket_end = clean.index("]") + 1
            clean = clean[bracket_end:].strip()
        idx = md_text.find(clean)
        if idx >= 0:
            return idx, idx + len(clean)
        snippet = clean[:120].strip()
        if len(snippet) > 20:
            idx = md_text.find(snippet)
            if idx >= 0:
                return idx, idx + len(snippet)
        for line in clean.split("\n"):
            line = line.strip()
            if len(line) > 25 and not line.startswith("[Description]") and not line.startswith("**["):
                idx = md_text.find(line)
                if idx >= 0:
                    return idx, idx + len(line)
        return -1, -1

    def _render_full_md_html(md_text: str, h_start: int, h_end: int) -> str:
        if h_start >= 0 and h_end > h_start:
            before = md_text[:h_start]
            target = md_text[h_start:h_end]
            after  = md_text[h_end:]
            marker_open  = '<span id="chunk-hl" class="hl-chunk">'
            marker_close = '</span>'
            marked = before + marker_open + target + marker_close + after
        else:
            marked = md_text
        body = _md2.markdown(
            marked,
            extras=["tables", "fenced-code-blocks", "header-ids", "break-on-newline"],
        )
        return body

    # ── Barre d'outils ──
    sources = col.distinct("source")
    if not sources:
        st.info("Aucun chunk en base. Lancez une ingestion dans l'onglet Pipeline RAG.")
    else:
        # Controles en ligne : source + recherche + type + bouton nettoyer
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([3, 3, 2, 1])
        with ctrl1:
            selected_source = st.selectbox(
                "Document", sources, key="source_tab3",
                label_visibility="collapsed",
            )
        with ctrl2:
            search = st.text_input("Rechercher...", key="search_tab3", label_visibility="collapsed",
                                   placeholder="Rechercher dans les chunks...")
        with ctrl3:
            type_filter = st.radio(
                "Type", ["Tous", "Chunks", "Tab/Fig", "Resumes"],
                horizontal=True, key="type_filter_tab3", label_visibility="collapsed",
            )
        with ctrl4:
            if st.button("Vider", key="clean_chunks_tab3", help="Supprimer tous les chunks"):
                try:
                    result = col.delete_many({})
                    st.success(f"{result.deleted_count} chunks supprimes")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        query_mongo = {"source": selected_source}
        if type_filter == "Chunks":
            query_mongo["chunk_type"] = {"$nin": ["summary"]}
        elif type_filter == "Resumes":
            query_mongo["chunk_type"] = "summary"
        elif type_filter == "Tab/Fig":
            query_mongo["chunk_type"] = {"$in": ["table", "figure", "mixed"]}
        if search:
            query_mongo["content"] = {"$regex": re.escape(search), "$options": "i"}

        chunks = list(col.find(query_mongo).sort([("section_idx", 1), ("chunk_idx", 1)]))
        normal_chunks = [c for c in chunks if c.get("chunk_type") != "summary"]
        summary_chunks = [c for c in chunks if c.get("chunk_type") == "summary"]
        n_tables = sum(1 for c in normal_chunks if c.get("chunk_type") in ("table", "figure", "mixed"))

        st.caption(
            f"{len(chunks)} chunks — "
            f"{len(normal_chunks)} texte/tableau"
            f"{f' (dont {n_tables} tab/fig)' if n_tables else ''}"
            f" — {len(summary_chunks)} resumes RAPTOR"
        )

        if not chunks:
            st.info("Aucun chunk pour ce document avec ces filtres.")
        else:
            if "sel_chunk_id" not in st.session_state:
                st.session_state["sel_chunk_id"] = None

            # ══════════════════════════════════════════════════════
            #  Chunks normaux — 2 colonnes
            # ══════════════════════════════════════════════════════
            if normal_chunks:
                col_left, col_right = st.columns([1, 2], gap="medium")

                # Trouver le chunk selectionne
                sel_id = st.session_state.get("sel_chunk_id")
                sel_chunk_data = None
                if sel_id is not None:
                    for c in normal_chunks:
                        if str(c.get("_id", "")) == sel_id:
                            sel_chunk_data = c
                            break

                # ═══════ COLONNE GAUCHE : liste + detail ═══════
                with col_left:
                    PAGE_SIZE = 25
                    total_normal = len(normal_chunks)
                    total_pages = max(1, (total_normal + PAGE_SIZE - 1) // PAGE_SIZE)

                    nav1, nav2 = st.columns([3, 1])
                    with nav2:
                        page_num = st.number_input(
                            "Page", min_value=1, max_value=total_pages,
                            value=1, step=1, key="chunk_page_input",
                            label_visibility="collapsed",
                        )
                    with nav1:
                        current_page = page_num - 1
                        page_start = current_page * PAGE_SIZE
                        page_end = min(page_start + PAGE_SIZE, total_normal)
                        st.caption(f"Chunks {page_start+1}-{page_end} / {total_normal}  (page {page_num}/{total_pages})")

                    for i in range(page_start, page_end):
                        chunk = normal_chunks[i]
                        cid = str(chunk.get("_id", i))
                        meta = {
                            "section_idx": chunk.get("section_idx", "?"),
                            "chunk_idx": chunk.get("chunk_idx", "?"),
                            "source": chunk.get("source", ""),
                            "page_number": chunk.get("page_number"),
                            "heading": chunk.get("heading", ""),
                            "breadcrumb": chunk.get("breadcrumb", ""),
                            "chunk_type": chunk.get("chunk_type", "chunk"),
                            "summary_num_chunks": chunk.get("summary_num_chunks"),
                        }
                        label = _chunk_display_label(meta)
                        is_selected = (sel_id == cid)
                        btn_type = "primary" if is_selected else "secondary"

                        if st.button(label, key=f"cb3_{cid}", width="stretch", type=btn_type):
                            st.session_state["sel_chunk_id"] = cid
                            st.rerun()

                    # ── Detail du chunk selectionne ──
                    if sel_chunk_data is not None:
                        st.markdown("---")
                        sel_heading = sel_chunk_data.get("heading", "") or sel_chunk_data.get("breadcrumb", "")
                        page_info = f" — p.{sel_chunk_data.get('page_number')}" if sel_chunk_data.get("page_number") else ""
                        st.markdown(f"**{sel_heading}{page_info}**")
                        sel_meta = {
                            "section_idx": sel_chunk_data.get("section_idx", "?"),
                            "chunk_idx": sel_chunk_data.get("chunk_idx", "?"),
                            "source": sel_chunk_data.get("source", ""),
                            "page_number": sel_chunk_data.get("page_number"),
                            "heading": sel_chunk_data.get("heading", ""),
                            "breadcrumb": sel_chunk_data.get("breadcrumb", ""),
                            "chunk_type": sel_chunk_data.get("chunk_type", "chunk"),
                            "summary_num_chunks": sel_chunk_data.get("summary_num_chunks"),
                            "keywords_str": sel_chunk_data.get("keywords_str", ""),
                            "questions_str": sel_chunk_data.get("questions_str", ""),
                            "keywords": sel_chunk_data.get("keywords", []),
                            "questions": sel_chunk_data.get("questions", []),
                            "table_description": sel_chunk_data.get("table_description", ""),
                            "has_table": sel_chunk_data.get("has_table", False),
                            "has_figure": sel_chunk_data.get("has_figure", False),
                            "entities": sel_chunk_data.get("entities", {}),
                            "entities_str": sel_chunk_data.get("entities_str", ""),
                            "entities_flat": sel_chunk_data.get("entities_flat", []),
                        }
                        _chunk_detail_block(sel_meta, sel_chunk_data.get("content", ""))

                # ═══════ COLONNE DROITE : markdown complet ═══════
                with col_right:
                    md_text = _load_md_text(selected_source)

                    if not md_text:
                        st.warning(f"Fichier markdown introuvable : {selected_source}")
                    else:
                        h_start, h_end = -1, -1
                        if sel_chunk_data is not None:
                            sel_content = sel_chunk_data.get("content", "")
                            h_start, h_end = _find_chunk_in_md(md_text, sel_content)

                        body_html = _render_full_md_html(md_text, h_start, h_end)

                        full_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
                        <style>
                          body {{
                            background: #0e1117; color: #d4d4d4;
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                            font-size: 14px; line-height: 1.7;
                            padding: 20px 24px; margin: 0;
                          }}
                          h1, h2, h3, h4 {{ color: #e0e0e0; margin-top: 1.2em; }}
                          h1 {{ font-size: 1.5em; border-bottom: 1px solid #333; padding-bottom: 6px; }}
                          h2 {{ font-size: 1.3em; }}
                          h3 {{ font-size: 1.1em; }}
                          a {{ color: #6daff7; }}
                          code {{ background: #1a1a2e; padding: 2px 5px; border-radius: 3px; font-size: 0.92em; }}
                          pre {{ background: #1a1a2e; padding: 12px; border-radius: 6px; overflow-x: auto; }}
                          table {{ border-collapse: collapse; margin: 12px 0; width: 100%; }}
                          th, td {{ border: 1px solid #444; padding: 6px 10px; text-align: left; }}
                          th {{ background: #1a1a2e; }}
                          .hl-chunk {{
                            background: linear-gradient(135deg, #2d1f00 0%, #3d2b00 100%);
                            color: #ffe066;
                            border-left: 4px solid #ff9800;
                            padding: 8px 12px; display: block;
                            margin: 8px 0; border-radius: 6px;
                            box-shadow: 0 0 12px rgba(255, 152, 0, 0.15);
                          }}
                        </style></head>
                        <body>{body_html}
                        <script>
                          var el = document.getElementById('chunk-hl');
                          if (el) {{ el.scrollIntoView({{ behavior: 'smooth', block: 'center' }}); }}
                        </script>
                        </body></html>"""
                        st.components.v1.html(full_html, height=900, scrolling=True)

            # ══════════════════════════════════════════════════════
            #  Resumes RAPTOR — expanders
            # ══════════════════════════════════════════════════════
            if summary_chunks:
                st.markdown("---")
                st.markdown(f"##### Resumes RAPTOR ({len(summary_chunks)})")
                for sc in summary_chunks:
                    sc_meta = {
                        "section_idx": sc.get("section_idx", "?"),
                        "chunk_idx": sc.get("chunk_idx", "?"),
                        "source": sc.get("source", ""),
                        "page_number": sc.get("page_number"),
                        "heading": sc.get("heading", ""),
                        "breadcrumb": sc.get("breadcrumb", ""),
                        "chunk_type": sc.get("chunk_type", "summary"),
                        "summary_num_chunks": sc.get("summary_num_chunks"),
                    }
                    label = _chunk_display_label(sc_meta)
                    with st.expander(label):
                        detail_meta = {
                            **sc_meta,
                            "keywords_str": sc.get("keywords_str", ""),
                            "questions_str": sc.get("questions_str", ""),
                            "keywords": sc.get("keywords", []),
                            "questions": sc.get("questions", []),
                            "table_description": sc.get("table_description", ""),
                            "has_table": sc.get("has_table", False),
                            "has_figure": sc.get("has_figure", False),
                            "entities": sc.get("entities", {}),
                            "entities_str": sc.get("entities_str", ""),
                            "entities_flat": sc.get("entities_flat", []),
                        }
                        _chunk_detail_block(detail_meta, sc.get("content", ""))

with tab3:
    st.title("Pipeline RAG")
    st.caption("Pipeline complete : chargement PDF, conversion Markdown, nettoyage, visualisation et ingestion RAG.")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PDF_DIR = PROJECT_ROOT / "Doc_Test_Pdf" / "Doc_Test_PDF"
    RAW_MD_DIR = PROJECT_ROOT / "docs" / "out"
    CLEAN_MD_DIR = PROJECT_ROOT / "data" / "out_clean_md"

    import base64 as _b64
    import markdown2 as _md2
    import time as _time

    # ─────────────────────────────────────────────────────────────────
    # ETAPE 1 — Charger un document
    # ─────────────────────────────────────────────────────────────────
    st.subheader("Etape 1 — Charger un document")

    source_mode = st.radio(
        "Source",
        ["Uploader un PDF", "Choisir un PDF existant", "Uploader un Markdown deja nettoye"],
        horizontal=True,
        key="pipeline_source_mode",
    )

    pdf_path_to_process: Path | None = None
    pdf_display_name: str = ""
    uploaded_md_content: str | None = None
    uploaded_md_name: str = ""

    if source_mode == "Uploader un PDF":
        uploaded_pdf = st.file_uploader("Choisissez un fichier PDF", type="pdf", key="pipeline_pdf_upload")
        if uploaded_pdf is not None:
            tmp_pdf = PROJECT_ROOT / "data" / uploaded_pdf.name
            tmp_pdf.parent.mkdir(parents=True, exist_ok=True)
            tmp_pdf.write_bytes(uploaded_pdf.getvalue())
            pdf_path_to_process = tmp_pdf
            pdf_display_name = uploaded_pdf.name

            # Apercu PDF
            with st.expander("Visualiser le PDF", expanded=False):
                zoom_level = st.slider("Zoom (%)", 50, 200, 100, step=10, key="pipeline_zoom")
                pdf_bytes = uploaded_pdf.getvalue()
                st.markdown(f"""
                <div style="display: flex; justify-content: center; width: 100%;">
                    <iframe src="data:application/pdf;base64,{_b64.b64encode(pdf_bytes).decode()}"
                            width="{int(800 * zoom_level / 100)}"
                            height="{int(1000 * zoom_level / 100)}"
                            frameborder="0"></iframe>
                </div>
                """, unsafe_allow_html=True)

    elif source_mode == "Choisir un PDF existant":
        existing_pdfs = sorted(PDF_DIR.glob("*.pdf")) if PDF_DIR.exists() else []
        if existing_pdfs:
            pdf_names = [p.name for p in existing_pdfs]
            selected_pdf = st.selectbox("PDF disponibles", pdf_names, key="pipeline_pdf_select")
            if selected_pdf:
                pdf_path_to_process = PDF_DIR / selected_pdf
                pdf_display_name = selected_pdf

                # Apercu PDF existant
                with st.expander("Visualiser le PDF", expanded=False):
                    zoom_level = st.slider("Zoom (%)", 50, 200, 100, step=10, key="pipeline_zoom_existing")
                    pdf_bytes = pdf_path_to_process.read_bytes()
                    st.markdown(f"""
                    <div style="display: flex; justify-content: center; width: 100%;">
                        <iframe src="data:application/pdf;base64,{_b64.b64encode(pdf_bytes).decode()}"
                                width="{int(800 * zoom_level / 100)}"
                                height="{int(1000 * zoom_level / 100)}"
                                frameborder="0"></iframe>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning(f"Aucun PDF trouve dans {PDF_DIR}")

    else:  # Uploader un Markdown
        uploaded_md = st.file_uploader("Choisissez un fichier .md", type="md", key="pipeline_md_upload")
        if uploaded_md is not None:
            uploaded_md_content = uploaded_md.getvalue().decode("utf-8")
            uploaded_md_name = uploaded_md.name

    # ─── Résumé de la sélection ───
    has_pdf = pdf_path_to_process is not None
    has_md = uploaded_md_content is not None

    if has_pdf:
        st.success(f"PDF selectionne : **{pdf_display_name}**")
    elif has_md:
        st.success(f"Markdown selectionne : **{uploaded_md_name}**")
    else:
        st.info("Selectionnez un document pour commencer.")
    if has_pdf or has_md:

        st.markdown("---")

        # ─────────────────────────────────────────────────────────────────
        # ETAPE 2 — Pre-processing (PDF uniquement)
        # ─────────────────────────────────────────────────────────────────
        if has_pdf:
            st.subheader("Etape 2 — Pre-processing : conversion + nettoyage")
            st.caption(
                "Conversion PDF vers Markdown via Docling avec marqueurs de page, "
                "puis nettoyage avance (headers/footers, parasites, fusion inter-pages, "
                "boilerplate, tables, references orphelines)."
            )

            if st.button("Lancer la conversion + nettoyage", width="stretch", type="primary", key="pipeline_btn_convert"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Conversion
                    status_text.info("Conversion PDF -> Markdown (Docling)...")
                    progress_bar.progress(10)
                    t0 = _time.time()

                    raw_md_path = pdf_to_md(str(pdf_path_to_process), out_dir=str(RAW_MD_DIR))
                    t_conv = _time.time() - t0
                    raw_text = Path(raw_md_path).read_text(encoding="utf-8")
                    raw_lines = raw_text.count("\n")

                    st.write(f"Conversion terminee en **{t_conv:.1f}s** — {raw_lines} lignes brutes")
                    progress_bar.progress(50)

                    # Nettoyage
                    status_text.info("Nettoyage avance du Markdown...")
                    t0 = _time.time()

                    stem = Path(pdf_path_to_process).stem
                    clean_path = str(CLEAN_MD_DIR / f"{stem}-clean.md")
                    CLEAN_MD_DIR.mkdir(parents=True, exist_ok=True)
                    cleaned_text = clean_md(raw_md_path, save_as=clean_path)
                    t_clean = _time.time() - t0
                    clean_lines = cleaned_text.count("\n") + 1 if cleaned_text else 0

                    progress_bar.progress(100)
                    status_text.success(
                        f"Pre-processing termine en **{t_conv + t_clean:.1f}s** — "
                        f"{raw_lines} -> {clean_lines} lignes"
                    )

                    # Sauvegarder dans la session pour les étapes suivantes
                    st.session_state.pipeline_clean_md_path = clean_path
                    st.session_state.pipeline_clean_md_text = cleaned_text
                    st.session_state.pipeline_source_name = pdf_display_name

                except Exception as e:
                    st.error(f"Erreur : {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    # Nettoyer le PDF temporaire si upload
                    if source_mode == "Uploader un PDF" and pdf_path_to_process.exists():
                        if pdf_path_to_process.parent == PROJECT_ROOT / "data":
                            pdf_path_to_process.unlink(missing_ok=True)

            # Afficher le MD nettoyé s'il existe en session
            if st.session_state.get("pipeline_clean_md_text"):
                with st.expander("Visualiser le Markdown nettoye", expanded=False):
                    html = _md2.markdown(
                        st.session_state.pipeline_clean_md_text,
                        extras=["tables", "fenced-code-blocks", "task_lists"],
                    )
                    st.markdown(f"""
                    <div style="
                        color: white;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.8;
                        padding: 20px;
                        border: 1px solid #1565c0;
                        border-radius: 6px;
                        background-color: rgba(0, 0, 0, 0.2);
                        max-height: 600px;
                        overflow-y: auto;
                    ">
                        {html}
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

        # ─── Si MD uploadé directement, stocker en session ───
        if has_md:
            st.session_state.pipeline_clean_md_text = uploaded_md_content
            st.session_state.pipeline_source_name = uploaded_md_name
            # Pas de chemin fichier encore, sera créé à l'ingestion
            st.session_state.pipeline_clean_md_path = None

            st.subheader("Etape 2 — Apercu du Markdown")
            with st.expander("Visualiser le Markdown", expanded=False):
                html = _md2.markdown(
                    uploaded_md_content,
                    extras=["tables", "fenced-code-blocks", "task_lists"],
                )
                st.markdown(f"""
                <div style="
                    color: white;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.8;
                    padding: 20px;
                    border: 1px solid #1565c0;
                    border-radius: 6px;
                    background-color: rgba(0, 0, 0, 0.2);
                    max-height: 600px;
                    overflow-y: auto;
                ">
                    {html}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")

        # ─────────────────────────────────────────────────────────────────
        # ETAPE 3 — Ingestion RAG
        # ─────────────────────────────────────────────────────────────────
        # Vérifier qu'on a un MD prêt (soit depuis conversion, soit uploadé)
        md_ready = bool(st.session_state.get("pipeline_clean_md_text"))

        st.subheader("Etape 3 — Ingestion dans le pipeline RAG")

        if not md_ready:
            st.info("Lancez d'abord l'etape 2 (conversion) ou uploadez un Markdown pour debloquer l'ingestion.")
        else:
            st.caption(
                "Chunking, enrichissement LLM (keywords + questions), resumes RAPTOR, "
                "embeddings, indexation ChromaDB, sauvegarde MongoDB et index BM25."
            )

            # ─── Options d'ingestion ───
            col_mode, col_raptor = st.columns(2)
            with col_mode:
                mode_options = list(CHUNKING_MODES.keys())
                current_mode_idx = mode_options.index(st.session_state.chunking_mode) if st.session_state.chunking_mode in mode_options else 0
                pipeline_chunking_mode = st.radio(
                    "Mode de chunking",
                    options=mode_options,
                    format_func=lambda m: f"**{m}** — {CHUNKING_MODES[m]}",
                    index=current_mode_idx,
                    key="pipeline_chunking_mode",
                    help="'technical' est recommande pour les documents ANSSI/CC. 'naive' pour les documents sans numerotation hierarchique.",
                )
            with col_raptor:
                pipeline_raptor = st.checkbox(
                    "Activer les resumes RAPTOR",
                    value=st.session_state.raptor_summaries,
                    key="pipeline_raptor",
                    help="Genere un chunk-resume par section de >=3 chunks. Ajoute ~3s par section eligible.",
                )

            col_kw, col_qq = st.columns(2)
            with col_kw:
                pipeline_keywords = st.slider(
                    "Mots-cles par chunk", 0, 10,
                    value=st.session_state.auto_keywords,
                    key="pipeline_keywords",
                    help="0 = desactive. Recommande : 5",
                )
            with col_qq:
                pipeline_questions = st.slider(
                    "Questions par chunk", 0, 5,
                    value=st.session_state.auto_questions,
                    key="pipeline_questions",
                    help="0 = desactive. Recommande : 3",
                )

            if pipeline_keywords > 0 or pipeline_questions > 0:
                st.info(
                    f"L'enrichissement ajoutera ~{2 * (pipeline_keywords + pipeline_questions)}s "
                    f"par chunk (appels LLM)."
                )

            # Synchroniser les choix vers session_state pour persistence et sauvegarde Settings
            st.session_state.chunking_mode = pipeline_chunking_mode
            st.session_state.raptor_summaries = pipeline_raptor
            st.session_state.auto_keywords = pipeline_keywords
            st.session_state.auto_questions = pipeline_questions

            st.markdown("---")

            # ─── Fonction d'ingestion en arriere-plan ───
            def _run_ingest_background(md_path, kw, qq, mode, raptor, ss):
                """Thread d'ingestion. Ecrit dans st.session_state (thread-safe)."""
                try:
                    def _bg_progress(step_name, pct):
                        ss.ingest_step = step_name
                        ss.ingest_progress = min(pct, 100)

                    stats = ingest_markdown(
                        md_path,
                        num_keywords=kw,
                        num_questions=qq,
                        chunking_mode=mode,
                        raptor_summaries=raptor,
                        progress_callback=_bg_progress,
                    )
                    ss.ingest_result = stats
                except Exception as e:
                    import traceback
                    ss.ingest_result = {"status": "error", "message": str(e), "traceback": traceback.format_exc()}
                finally:
                    ss.ingest_running = False

            # ─── Etat de l'ingestion ───
            is_running = st.session_state.ingest_running
            has_result = st.session_state.ingest_result is not None and not is_running

            if is_running:
                # ── Ingestion en cours : progression detaillee ──
                pct = st.session_state.ingest_progress
                step = st.session_state.ingest_step

                # Parser current/total depuis le step (ex: "Enrichissement chunk 280/645...")
                import re as _re
                _m = _re.search(r'(\d+)\s*/\s*(\d+)', step)
                sub_current, sub_total = (int(_m.group(1)), int(_m.group(2))) if _m else (0, 0)

                if sub_total > 0:
                    # ---- Affichage simple du compteur pour l'etape courante ----
                    st.markdown(f"""
                    <div style="
                        background: #0d1117; border: 1px solid #42a5f5;
                        border-radius: 8px; padding: 1rem 1.5rem; margin: 0.5rem 0;
                        display: flex; justify-content: space-between; align-items: center;
                    ">
                      <span style="color:#90caf9; font-weight:600; font-size:1.1rem;">
                        {step.split(' ')[0]} en cours...
                      </span>
                      <span style="
                        color:#fff; font-weight:700; font-size:1.4rem;
                        font-variant-numeric: tabular-nums;
                        background: #1a237e; padding: 0.3rem 0.8rem; border-radius: 6px;
                        border: 1px solid #1e88e5;
                      ">{sub_current} / {sub_total}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Etape sans compteur (vocabulaire, embeddings, etc.)
                    st.markdown(
                        f'<div style="background:#0d1117; border:1px solid #42a5f5; '
                        f'border-radius:8px; padding:1rem 1.5rem; margin:0.5rem 0; '
                        f'color:#90caf9; font-weight:600; font-size:1.1rem;">'
                        f'{step}</div>',
                        unsafe_allow_html=True,
                    )

                st.caption("Vous pouvez naviguer librement dans les autres onglets -- "
                           "l'ingestion continue en arriere-plan.")

            elif has_result:
                # ── Ingestion terminee : afficher les resultats ──
                stats = st.session_state.ingest_result

                if stats.get("status") == "success":
                    st.success("Ingestion terminee avec succes !")

                    # Metriques
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Chunks crees", stats["num_chunks"])
                    with col2:
                        st.metric("Vocabulaire", stats["vocab_size"])
                    with col3:
                        st.metric("Acronymes", stats["acronyms_count"])
                    with col4:
                        st.metric("Embeddings", stats["embeddings_count"])

                    st.info(f"Mode de chunking : **{stats.get('chunking_mode', 'naive')}**")

                    # Enrichissement
                    enh = stats.get("enhancement", {})
                    if enh.get("num_keywords", 0) > 0 or enh.get("num_questions", 0) > 0:
                        col_e1, col_e2, col_e3 = st.columns(3)
                        with col_e1:
                            st.metric("Chunks enrichis", enh.get("chunks_enhanced", 0))
                        with col_e2:
                            st.metric("Keywords/chunk", enh.get("num_keywords", 0))
                        with col_e3:
                            st.metric("Questions/chunk", enh.get("num_questions", 0))
                        table_descs = enh.get("table_descriptions", 0)
                        if table_descs > 0:
                            st.success(f"Descriptions de tableaux generees : **{table_descs}**")
                        ner_count = enh.get("chunks_with_entities", 0)
                        if ner_count > 0:
                            st.success(f"Chunks avec entites nommees : **{ner_count}**")

                    # RAPTOR
                    raptor_stats = stats.get("raptor", {})
                    if raptor_stats.get("enabled"):
                        st.metric("Resumes RAPTOR generes", raptor_stats.get("summaries_generated", 0))

                    # GraphRAG
                    graph_st = stats.get("graph", {})
                    if graph_st.get("nodes", 0) > 0:
                        col_g1, col_g2 = st.columns(2)
                        with col_g1:
                            st.metric("Entites (graphe)", graph_st["nodes"])
                        with col_g2:
                            st.metric("Relations (graphe)", graph_st["edges"])
                        top_ents = graph_st.get("top_entities", [])
                        if top_ents:
                            with st.expander("Top entites du graphe"):
                                for ent, deg in top_ents[:10]:
                                    st.write(f"  **{ent}** -- {deg} connexions")

                    # Chunks par section
                    with st.expander("Chunks par section"):
                        for sec, count in stats.get("chunks_per_section", {}).items():
                            st.write(f"  Section {sec}: {count} chunks")

                    st.info("Rendez-vous dans l'onglet 'Exploration chunks' pour voir les nouveaux chunks ingeres.")

                    # Bouton pour relancer une nouvelle ingestion
                    if st.button("Nouvelle ingestion", key="pipeline_btn_new_ingest"):
                        st.session_state.ingest_result = None
                        st.rerun()

                elif stats.get("status") == "error":
                    st.error(f"Erreur lors de l'ingestion : {stats['message']}")
                    if stats.get("traceback"):
                        st.code(stats["traceback"])
                    if st.button("Recommencer", key="pipeline_btn_retry_ingest"):
                        st.session_state.ingest_result = None
                        st.rerun()
                else:
                    st.error(f"Erreur lors de l'ingestion : {stats.get('message', 'Erreur inconnue')}")
                    if st.button("Recommencer", key="pipeline_btn_retry_ingest2"):
                        st.session_state.ingest_result = None
                        st.rerun()

            else:
                # ── Pas d'ingestion en cours : afficher le bouton ──
                if st.button("Lancer l'ingestion RAG", width="stretch", type="primary", key="pipeline_btn_ingest"):
                    # Preparer le fichier MD sur disque
                    clean_md_path = st.session_state.get("pipeline_clean_md_path")
                    tmp_md_created = False

                    if clean_md_path is None or not Path(clean_md_path).exists():
                        tmp_md_path = PROJECT_ROOT / "data" / st.session_state.pipeline_source_name
                        tmp_md_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp_md_path.write_text(st.session_state.pipeline_clean_md_text, encoding="utf-8")
                        clean_md_path = str(tmp_md_path)

                    # Reinitialiser l'etat
                    st.session_state.ingest_running = True
                    st.session_state.ingest_progress = 0
                    st.session_state.ingest_step = "Demarrage..."
                    st.session_state.ingest_result = None

                    # Lancer le thread
                    t = threading.Thread(
                        target=_run_ingest_background,
                        args=(clean_md_path, pipeline_keywords, pipeline_questions,
                              pipeline_chunking_mode, pipeline_raptor, st.session_state),
                        daemon=True,
                    )
                    t.start()
                    st.session_state.ingest_thread = t
                    st.rerun()

with tab4:
    st.title("Système Prompt")
    st.write("Modifiez le système prompt utilisé pour générer les réponses du LLM.")
    
    # Afficher le prompt actuel
    st.subheader("Prompt actuel")
    current_prompt = st.text_area(
        "Système Prompt :",
        value=st.session_state.system_prompt,
        height=400,
        key="prompt_editor"
    )

    # Boutons d'action
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Sauvegarder"):
            st.session_state.system_prompt = current_prompt
            st.success("Système prompt mis à jour !")
    
    with col2:
        if st.button("Réinitialiser"):
            st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
            st.info("Système prompt réinitialisé au default.")
    
    with col3:
        if st.button("Copier"):
            st.write("Prompt copié dans le presse-papiers (voir le texte ci-dessous)")
            st.code(st.session_state.system_prompt)
    
    st.subheader("Prévisualisation")
    st.info(f"Longueur du prompt : {len(st.session_state.system_prompt)} caractères")

# ---------------------------------------------------------------------- Onglet Settings ----------------------------------------------------------------------#
import subprocess
import os
import sys
# Fonction pour redémarrer l'application Streamlit avec la venv activée
def restart_app():
    """Redémarre l'application Streamlit avec la venv activée."""
    try:
        project_root = str(Path(__file__).resolve().parent.parent)
        # Chemin vers le script de la venv
        venv_activate = os.path.join(project_root, "rag_venv", "bin", "activate")
        
        # Vérifier que la venv existe
        if not os.path.exists(venv_activate):
            return False, f"La venv n'existe pas à : {venv_activate}"
        
        # Vérifier qu'on est dans la venv
        if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            return False, "Vous n'êtes pas dans la venv"
        
        # Redémarrer Streamlit
        app_path = os.path.join(project_root, "app", "streamlit_app.py")
        os.execvp("streamlit", ["streamlit", "run", app_path])
        return True, "Application redémarrée"

    except Exception as e:
        return False, f"Erreur : {str(e)}"

# Fonction pour récupérer les modèles Ollama disponibles
def get_ollama_models():
    """Retourne la liste des modèles Ollama disponibles."""
    try:
        res = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        lines = res.stdout.splitlines()
        models = []
        for line in lines:
            line = line.strip()
            if not line or line.lower().startswith("name") or line.startswith("---"):
                continue
            parts = line.split()
            if parts:
                models.append(parts[0])
        return list(dict.fromkeys(models))
    except Exception:
        return []


# Modèles d'embedding connus
EMBEDDING_MODELS = ["qwen3-embedding:latest","bge-m3:567m", "nomic-embed-text:latest"]

# Modèles LLM connus
LLM_MODELS = ["qwen3-next:latest","magistral:latest", "gpt-oss:latest","qwen3-vl:8b", "qwen3-vl:2b", "llama3.3:latest", "llama3.1:latest", "deepseek-r1:8b", "codegemma:latest"]

def filter_embedding_models(all_models):
    """Filtre les modèles d'embedding parmi tous les modèles."""
    return [m for m in all_models if any(emb in m.lower() for emb in EMBEDDING_MODELS)]

def filter_llm_models(all_models):
    """Filtre les modèles LLM parmi tous les modèles."""
    return [m for m in all_models if any(llm in m.lower() for llm in LLM_MODELS)]

with tab5:
    st.title("Settings")
    st.markdown("---")
    
    # Récupérer les modèles disponibles
    all_models = get_ollama_models()
    if not all_models:
        st.warning(" Aucun modèle Ollama détecté. Assurez-vous que Ollama est installé et en cours d'exécution.")
        all_models = [st.session_state.embed_model, st.session_state.gen_model]

    else:
        # Filtrer les modèles par type
        embed_models = filter_embedding_models(all_models)
        llm_models = filter_llm_models(all_models)
        
        if not embed_models:
            st.warning("Aucun modèle d'embedding détecté.")
            embed_models = [st.session_state.embed_model]
        
        if not llm_models:
            st.warning("Aucun modèle LLM détecté.")
            llm_models = [st.session_state.gen_model]

    # NUM_CHUNKS
    st.subheader(" Nombre de chunks à récupérer dans \"Classement final\"")
    new_num = st.slider("Nombre de chunks", 1, 50, value=st.session_state.num_chunks, step=1, label_visibility="collapsed")

    # Embedding model
    st.subheader(" Modèle d'embedding")
    selected_embed = st.selectbox(
        "Choisir un modèle",
        embed_models,
        index=embed_models.index(st.session_state.embed_model) if st.session_state.embed_model in embed_models else 0,
        key="embed_select"
    )
    st.info(f" Modèles détectés : {', '.join(embed_models)}")

    # LLM model
    st.subheader(" Modèle LLM")
    selected_gen = st.selectbox(
        "Choisir un modèle",
        llm_models,
        index=llm_models.index(st.session_state.gen_model) if st.session_state.gen_model in llm_models else 0,
        key="gen_select"
    )
    st.info(f" Modèles détectés : {', '.join(llm_models)}")

    # Configuration des poids RRF
    st.markdown("---")
    st.subheader(" Poids de fusion RRF")

    from config import WEIGHT_SEMANTIC, WEIGHT_BM25
    new_weight_semantic = st.slider("Poids attribué à la recherche sémantique (semantic search)", 0.0, 1.0, value=WEIGHT_SEMANTIC, step=0.1)
    new_weight_bm25 = st.slider("Poids attribué à la recherche BM25 (keyword search)", 0.0, 1.0, value=WEIGHT_BM25, step=0.1)

    st.session_state.weight_semantic = new_weight_semantic
    st.session_state.weight_bm25 = new_weight_bm25

    # ── Self-RAG : paramètres fins (le ON/OFF est dans la sidebar de tab1) ──
    st.markdown("---")
    st.subheader("Détection hors-scope")
    st.caption(
        "Si le meilleur score du cross-encoder est inférieur à ce seuil, "
        "le RAG répond qu'il n'a pas d'information pertinente — sans appeler le LLM de génération."
    )
    new_ce_relevance_threshold = st.slider(
        "Seuil CE relevance (max score chunks)",
        min_value=0.50, max_value=0.60,
        value=float(st.session_state.ce_relevance_threshold),
        step=0.005,
        format="%.3f",
        help="Score CE sigmoïde. Neutre = 0.500. In-scope typique ≥ 0.530. "
             "Recommandé : 0.525. Augmenter si trop de faux positifs hors-scope.",
    )

    st.markdown("---")
    st.subheader("Self-RAG — paramètres fins")
    st.caption(
        "Le Self-RAG s'active/désactive depuis la sidebar de l'onglet SSH GPT. "
        "Ces paramètres règlent le comportement quand il est actif."
    )
    new_self_rag_threshold = st.slider(
        "Seuil de score (en dessous = retry)",
        min_value=0.1, max_value=0.9,
        value=float(st.session_state.self_rag_threshold),
        step=0.05,
        help="Score moyen pondéré (Fidélité×0.45 + Pertinence réponse×0.25 + Pertinence contexte×0.30). "
             "En dessous de ce seuil, le pipeline retente. Recommandé : 0.55.",
    )
    new_self_rag_retries = st.number_input(
        "Nombre maximum de retries",
        min_value=1, max_value=3,
        value=int(st.session_state.self_rag_max_retries),
        step=1,
        help="1 = 2 essais au total (initial + 1 retry). Recommandé : 1.",
    )
    new_self_rag_enabled = st.session_state.self_rag_enabled  # valeur portée par la sidebar

    # Bouton unique pour enregistrer
    if st.button(" Enregistrer dans config.py", width="stretch"):
        try:
            p = Path(__file__).resolve().parent.parent / "config.py"
            text = p.read_text()
            text = re.sub(r"NUM_CHUNKS\s*=\s*\d+", f"NUM_CHUNKS = {new_num}", text)
            text = re.sub(r"EMBED_MODEL\s*=\s*['\"].*?['\"]", f'EMBED_MODEL = "{selected_embed}"', text)
            text = re.sub(r"GEN_MODEL\s*=\s*['\"].*?['\"]", f'GEN_MODEL = "{selected_gen}"', text)
            text = re.sub(r"WEIGHT_SEMANTIC\s*=\s*[\d.]+", f"WEIGHT_SEMANTIC = {new_weight_semantic}", text)
            text = re.sub(r"WEIGHT_BM25\s*=\s*[\d.]+", f"WEIGHT_BM25 = {new_weight_bm25}", text)
            text = re.sub(r"AUTO_KEYWORDS\s*=\s*\d+", f"AUTO_KEYWORDS = {st.session_state.auto_keywords}", text)
            text = re.sub(r"AUTO_QUESTIONS\s*=\s*\d+", f"AUTO_QUESTIONS = {st.session_state.auto_questions}", text)
            text = re.sub(r"RAPTOR_SUMMARIES\s*=\s*(True|False)", f"RAPTOR_SUMMARIES = {st.session_state.raptor_summaries}", text)
            text = re.sub(r"CHUNKING_MODE\s*=\s*['\"].*?['\"]", f'CHUNKING_MODE = "{st.session_state.chunking_mode}"', text)
            text = re.sub(r"SELF_RAG_ENABLED\s*=\s*(True|False)", f"SELF_RAG_ENABLED = {new_self_rag_enabled}", text)
            text = re.sub(r"SELF_RAG_THRESHOLD\s*=\s*[\d.]+", f"SELF_RAG_THRESHOLD = {new_self_rag_threshold}", text)
            text = re.sub(r"SELF_RAG_MAX_RETRIES\s*=\s*\d+", f"SELF_RAG_MAX_RETRIES = {int(new_self_rag_retries)}", text)
            text = re.sub(r"CE_RELEVANCE_THRESHOLD\s*=\s*[\d.]+", f"CE_RELEVANCE_THRESHOLD = {new_ce_relevance_threshold}", text)
            p.write_text(text)
            st.success(" config.py mis à jour. Redémarrez l'application.")
            st.session_state.num_chunks = new_num
            st.session_state.embed_model = selected_embed
            st.session_state.gen_model = selected_gen
            st.session_state.ce_relevance_threshold = new_ce_relevance_threshold

            # message pour proposer le redémarrage
            st.info("Les changements ont été enregistrés. Cliquez sur 'Redémarrer l'app' ci-dessous pour appliquer les modifications.")
        except Exception as e:
            st.error(f"Erreur : {e}")
# Bouton pour redémarrer l'application
    st.markdown("---")  
    
    if st.button("Redémarrer l'app", width="stretch"):
        success, message = restart_app()
        if success:
            st.success(message)
        else:
            st.error(message)
            st.stop()
# Section pour la mise à jour d'Ollama
    st.markdown("---")
    st.subheader("Mise à jour d'Ollama")

    if st.button("Mettre à jour Ollama", width="stretch"):
        try:
            st.info("Installation/mise à jour d'Ollama...")
            result = subprocess.run(
                ["bash", "-c", "curl -fsSL https://ollama.ai/install.sh | sh"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                st.success("Ollama est à jour !")
                st.write(result.stdout)
            else:
                st.warning(result.stderr)
        except subprocess.TimeoutExpired:
            st.error("La mise à jour a pris trop de temps (timeout)")
        except Exception as e:
            st.error(f"Erreur : {str(e)}")

    if st.button("Vérifier la version d'Ollama", width="stretch"):
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=10)
            st.code(result.stdout)
        except Exception as e:
            st.error(f"Erreur : {str(e)}")

# ══════════════════════════════════════════════════════════════════════
#  TAB 6 — Graphe d'entites
# ══════════════════════════════════════════════════════════════════════
with tab6:
    st.title("Graphe d'entites")
    st.caption("Visualisation du graphe de connaissances construit par GraphRAG a partir des entites nommees extraites des chunks.")

    from nlp.graph_builder import load_graph_from_mongo, graph_stats
    import json as _json_tab8
    from pymongo import MongoClient

    # Récupérer la liste des documents disponibles
    @st.cache_data(ttl=60)
    def get_available_documents():
        try:
            client = MongoClient("mongodb://localhost:27017")
            db = client["ragdb"]
            col = db["entity_graph"]
            docs = col.distinct("source_doc")
            return docs if docs else []
        except Exception as e:
            st.error(f"Erreur lors de la récupération des documents : {e}")
            return []

    available_docs = get_available_documents()

    if not available_docs:
        st.info("Aucun graphe d'entites en base. Lancez une ingestion avec GraphRAG active dans l'onglet Pipeline RAG.")
    else:
        selected_doc = st.selectbox("Sélectionnez un document pour voir son graphe :", available_docs)

        # Charger le graphe
        @st.cache_resource(ttl=300)
        def _load_graph_cached(doc_name):
            try:
                return load_graph_from_mongo(source_doc=doc_name)
            except Exception as e:
                st.error(f"Erreur chargement du graphe : {e}")
                return None

        G = _load_graph_cached(selected_doc)

        if G is None or G.number_of_nodes() == 0:
            st.info(f"Aucun graphe d'entites trouvé pour le document {selected_doc}.")
        else:
            g_stats = graph_stats(G)

            # ── Stats du graphe ──
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Entites", g_stats["nodes"])
            with m2:
                st.metric("Relations", g_stats["edges"])
            with m3:
                n_typed = sum(v for k, v in g_stats.get("relation_types", {}).items() if k != "co_occurrence")
                st.metric("Relations typees", n_typed)
            with m4:
                n_cooc = g_stats.get("relation_types", {}).get("co_occurrence", 0)
                st.metric("Co-occurrences", n_cooc)

            # ── Sous-tabs : Visualisation / Exploration / Top entites ──
            graph_sub1, graph_sub2, graph_sub3 = st.tabs(["Visualisation", "Explorer une entite", "Top entites"])

            # ═══════ Sous-tab 1 : Visualisation interactive vis.js ═══════
            with graph_sub1:
                st.markdown("##### Graphe interactif")
                st.caption("Cliquez-glissez pour naviguer. Scroll pour zoomer. Les noeuds sont colores par type d'entite.")

                # Construire les donnees vis.js
                TYPE_COLORS = {
                    "ORG": "#4fc3f7",
                    "PER": "#81c784",
                    "LOC": "#ffb74d",
                    "NORM": "#ce93d8",
                    "ACRO": "#fff176",
                    "MISC": "#90a4ae",
                    "UNKNOWN": "#616161",
                }

                # Limiter a max_nodes pour la perf
                max_vis_nodes = st.slider("Nombre max de noeuds a afficher", 20, 300, 80, step=10, key="graph_max_nodes")
                top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_vis_nodes]
                vis_node_set = set(n for n, _ in top_nodes)

                vis_nodes = []
                for node, deg in top_nodes:
                    ndata = G.nodes.get(node, {})
                    ntype = ndata.get("type", "UNKNOWN")
                    color = TYPE_COLORS.get(ntype, "#616161")
                    size = min(8 + deg * 2, 50)
                    vis_nodes.append({
                        "id": node,
                        "label": node,
                        "color": color,
                        "size": size,
                        "title": f"{node} ({ntype}) — {deg} connexions",
                        "font": {"color": "#e0e0e0", "size": 11},
                    })

                vis_edges = []
                for src, tgt, edata in G.edges(data=True):
                    if src in vis_node_set and tgt in vis_node_set:
                        rel = edata.get("relation", "co_occurrence")
                        w = edata.get("weight", 1)
                        is_typed = rel != "co_occurrence"
                        vis_edges.append({
                            "from": src,
                            "to": tgt,
                            "label": rel if is_typed else "",
                            "color": {"color": "#ff9800" if is_typed else "#555555", "opacity": 0.7},
                            "width": min(1 + w, 5),
                            "arrows": "to" if is_typed else "",
                            "font": {"color": "#ff9800", "size": 9, "strokeWidth": 0},
                        })

                nodes_json = _json_tab8.dumps(vis_nodes)
                edges_json = _json_tab8.dumps(vis_edges)

                # Legende
                legend_items = " ".join(
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'background:{c};border-radius:50%;margin-right:4px;vertical-align:middle;"></span>'
                    f'<span style="color:#ccc;margin-right:14px;font-size:12px;">{t}</span>'
                    for t, c in TYPE_COLORS.items()
                    if t in g_stats.get("node_types", {})
                )

                vis_html = f"""<!DOCTYPE html><html><head>
                <meta charset="utf-8">
                <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                <style>
                  body {{ margin: 0; padding: 0; background: #0e1117; }}
                  #graph-container {{ width: 100%; height: 680px; border: 1px solid #333; border-radius: 8px; }}
                  #legend {{ padding: 8px 12px; }}
                </style>
                </head><body>
                <div id="legend">{legend_items}</div>
                <div id="graph-container"></div>
                <script>
                  var nodes = new vis.DataSet({nodes_json});
                  var edges = new vis.DataSet({edges_json});
                  var container = document.getElementById('graph-container');
                  var data = {{ nodes: nodes, edges: edges }};
                  var options = {{
                    physics: {{
                      solver: 'forceAtlas2Based',
                      forceAtlas2Based: {{ gravitationalConstant: -40, centralGravity: 0.005, springLength: 120 }},
                      stabilization: {{ iterations: 150 }}
                    }},
                    interaction: {{ hover: true, tooltipDelay: 100, zoomView: true, dragView: true }},
                    layout: {{ improvedLayout: true }},
                  }};
                  var network = new vis.Network(container, data, options);
                </script>
                </body></html>"""

                st.components.v1.html(vis_html, height=740, scrolling=False)

            # ═══════ Sous-tab 2 : Explorer une entite ═══════
            with graph_sub2:
                st.markdown("##### Rechercher une entite")
                all_entities = sorted(G.nodes())
                search_entity = st.selectbox(
                    "Entite",
                    [""] + all_entities,
                    key="graph_search_entity",
                    format_func=lambda x: "Choisir une entite..." if x == "" else x,
                )

                if search_entity and search_entity in G.nodes():
                    ndata = G.nodes[search_entity]
                    etype = ndata.get("type", "UNKNOWN")
                    ecount = ndata.get("count", 0)
                    echunks = ndata.get("chunk_ids", set())

                    # Info entite
                    info1, info2, info3 = st.columns(3)
                    with info1:
                        st.metric("Type", etype)
                    with info2:
                        st.metric("Apparitions", ecount)
                    with info3:
                        st.metric("Chunks", len(echunks))

                    # Voisins
                    st.markdown("##### Relations")
                    successors = list(G.successors(search_entity))
                    predecessors = list(G.predecessors(search_entity))
                    all_neighbors = set(successors + predecessors)

                    if all_neighbors:
                        rows = []
                        for neighbor in sorted(all_neighbors):
                            # Relation sortante
                            if G.has_edge(search_entity, neighbor):
                                ed = G[search_entity][neighbor]
                                rows.append({
                                    "Direction": "->",
                                    "Entite liee": neighbor,
                                    "Type entite": G.nodes.get(neighbor, {}).get("type", "?"),
                                    "Relation": ed.get("relation", "co_occurrence"),
                                    "Poids": ed.get("weight", 1),
                                    "Chunks communs": len(ed.get("chunk_ids", set())),
                                })
                            # Relation entrante
                            if G.has_edge(neighbor, search_entity):
                                ed = G[neighbor][search_entity]
                                rows.append({
                                    "Direction": "<-",
                                    "Entite liee": neighbor,
                                    "Type entite": G.nodes.get(neighbor, {}).get("type", "?"),
                                    "Relation": ed.get("relation", "co_occurrence"),
                                    "Poids": ed.get("weight", 1),
                                    "Chunks communs": len(ed.get("chunk_ids", set())),
                                })

                        import pandas as _pd_tab8
                        df = _pd_tab8.DataFrame(rows)
                        st.dataframe(df, width="stretch", hide_index=True)
                    else:
                        st.info("Cette entite n'a aucune relation dans le graphe.")

                    # Chunks associes
                    if echunks:
                        with st.expander(f"Chunks contenant '{search_entity}' ({len(echunks)})"):
                            for cid in sorted(echunks):
                                chunk_doc = col.find_one({"_id": cid})
                                if chunk_doc:
                                    heading = chunk_doc.get("heading", "") or chunk_doc.get("breadcrumb", "")
                                    page = chunk_doc.get("page_number", "")
                                    preview = (chunk_doc.get("content", "")[:150] + "...") if chunk_doc.get("content") else ""
                                    st.markdown(f"**{cid}** — {heading} (p.{page})")
                                    st.caption(preview)
                                else:
                                    st.caption(f"{cid} (chunk non trouve en base)")

            # ═══════ Sous-tab 3 : Top entites ═══════
            with graph_sub3:
                st.markdown("##### Entites les plus connectees")

                top_n = st.slider("Nombre d'entites", 10, 50, 20, key="graph_top_n")
                top = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_n]

                import pandas as _pd_tab8b
                top_rows = []
                for entity, degree in top:
                    ndata = G.nodes.get(entity, {})
                    top_rows.append({
                        "Entite": entity,
                        "Type": ndata.get("type", "UNKNOWN"),
                        "Connexions": degree,
                        "Apparitions": ndata.get("count", 0),
                        "Chunks": len(ndata.get("chunk_ids", set())),
                    })
                df_top = _pd_tab8b.DataFrame(top_rows)
                st.dataframe(df_top, width="stretch", hide_index=True)

                # Distribution par type
                st.markdown("##### Repartition par type d'entite")
                type_counts = g_stats.get("node_types", {})
                if type_counts:
                    df_types = _pd_tab8b.DataFrame(
                        [{"Type": t, "Nombre": n} for t, n in sorted(type_counts.items(), key=lambda x: -x[1])]
                    )
                    st.bar_chart(df_types, x="Type", y="Nombre")

                # Distribution des relations
                rel_counts = g_stats.get("relation_types", {})
                if rel_counts:
                    st.markdown("##### Types de relations")
                    df_rels = _pd_tab8b.DataFrame(
                        [{"Relation": r, "Nombre": n} for r, n in sorted(rel_counts.items(), key=lambda x: -x[1])]
                    )
                    st.dataframe(df_rels, width="stretch", hide_index=True)
