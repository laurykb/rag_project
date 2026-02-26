# preprocessing/pdf_to_markdown.py
# Pipeline PDF -> Markdown avec nettoyage avance
# Ameliorations inspirees RAGFlow DeepDoc :
#   1. Suppression automatique headers/footers repetes (detection statistique)
#   2. Nettoyage lignes parasites (numeros de page, lignes courtes, copyrights)
#   3. Fusion paragraphes coupes entre pages
#   4. Suppression sections boilerplate generiques (TOC, TOF, TOT)
#   5. Validation et reparation tableaux Markdown
#   6. Marquage references orphelines (figures sans image)

from pathlib import Path
from docling.document_converter import DocumentConverter
from collections import Counter
import re
import os
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Extraction du numero de page d'un DocItem via sa provenance
# ═══════════════════════════════════════════════════════════════════════
def _item_page(item) -> int | None:
    """Renvoie le numero de page (1-based) d'un DocItem Docling, ou None."""
    try:
        if hasattr(item, "prov") and item.prov:
            return item.prov[0].page_no
    except Exception:
        pass
    return None


def _item_text(item, doc) -> str:
    """Extrait le texte/markdown d'un DocItem Docling (compatible v2.48+)."""
    type_name = type(item).__name__

    # TableItem : export markdown (avec doc en arg)
    if type_name == "TableItem":
        try:
            return item.export_to_markdown(doc=doc)
        except Exception:
            return getattr(item, "text", "") or ""

    # PictureItem : marquer comme image manquante (sera traite en etape 5)
    if type_name == "PictureItem":
        caption = ""
        try:
            caption = item.caption_text(doc) or ""
        except Exception:
            pass
        if caption.strip():
            return f"[Figure: {caption.strip()}]"
        return "<!-- image -->"

    # SectionHeaderItem : construire le heading markdown
    if type_name == "SectionHeaderItem":
        text = getattr(item, "text", "") or ""
        return text  # le level sera utilise par l'appelant

    # ListItem : prefixe tiret
    if type_name == "ListItem":
        text = getattr(item, "text", "") or ""
        return f"- {text}" if text.strip() else ""

    # TextItem et autres
    return getattr(item, "text", "") or ""


# ═══════════════════════════════════════════════════════════════════════
# ETAPE 1 : Conversion PDF -> Markdown brut avec marqueurs de page
# ═══════════════════════════════════════════════════════════════════════
def pdf_to_md(src_pdf: str, out_dir: str = "./out") -> str:
    """
    Convertit un PDF en Markdown via Docling avec :
    - Marqueurs de page <!-- page:N -->
    - Headings Markdown corrects (## pour les titres)
    - Export explicite par type d'item (compatible Docling v2.48+)
    """
    src = Path(src_pdf)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_md = out / f"{src.stem}.md"

    log.info(f"[pdf_to_md] Conversion de {src.name}...")
    conv = DocumentConverter().convert(str(src))
    doc = conv.document

    md_lines: list[str] = []
    current_page: int | None = None

    for item, level in doc.iterate_items():
        page = _item_page(item)

        # Inserer un marqueur de page a chaque changement
        if page is not None and page != current_page:
            md_lines.append(f"\n<!-- page:{page} -->\n")
            current_page = page

        type_name = type(item).__name__
        text = _item_text(item, doc)

        if not text or not text.strip():
            continue

        # Pour les SectionHeaderItem : construire le heading markdown
        if type_name == "SectionHeaderItem":
            # Determiner la profondeur via la numerotation (1. / 1.2. / 1.2.3.)
            depth = _heading_depth_from_text(text)
            prefix = "#" * max(1, min(depth, 6))
            # Eviter le double prefixe si le texte commence deja par #
            if text.strip().startswith("#"):
                md_lines.append(text.strip())
            else:
                md_lines.append(f"{prefix} {text.strip()}")
        else:
            md_lines.append(text)

    md_text = "\n\n".join(md_lines)

    # Verification : si le markdown est quasi vide (PDF scanne sans texte)
    text_only = re.sub(r"<!--.*?-->", "", md_text).strip()
    text_only = re.sub(r"\[Figure:.*?\]", "", text_only).strip()
    if len(text_only) < 100:
        log.warning(f"[pdf_to_md] Tres peu de texte extrait ({len(text_only)} chars) "
                     f"- le PDF est probablement scanne (images). "
                     f"Un OCR externe serait necessaire.")

    out_md.write_text(md_text, encoding="utf-8")
    log.info(f"[pdf_to_md] MD brut ecrit -> {out_md} ({len(md_lines)} blocs)")
    return str(out_md)


def _heading_depth_from_text(text: str) -> int:
    """
    Determine la profondeur d'un heading a partir de sa numerotation.
    '1. Introduction' -> 1 (=> ## )
    '2.2.1 TOE' -> 3 (=> #### )
    'AVERTISSEMENT' -> 1 (=> ## )
    """
    text = text.strip().lstrip("#").strip()
    m = re.match(r"^(\d+(?:\.\d+)*)\s*\.?\s+", text)
    if m:
        parts = [p for p in m.group(1).split(".") if p]
        return min(len(parts) + 1, 6)  # +1 car # est reserve au titre du doc
    return 2  # par defaut, ## pour les titres non numerotes


# ═══════════════════════════════════════════════════════════════════════
# ETAPE 2 : Nettoyage avance du Markdown
# ═══════════════════════════════════════════════════════════════════════

# ---- Amelioration 1 : Detection statistique des headers/footers ------
def _detect_repeated_blocks(lines: list[str], min_length: int = 25,
                            min_ratio: float = 0.08) -> set[str]:
    """
    Detecte les blocs de texte repetes sur de nombreuses pages.
    Un texte apparaissant sur >8% des pages (et au moins 3 fois)
    est considere comme un header/footer recurrent.

    Retourne un set de textes normalises a supprimer.
    """
    # Compter les pages totales (via marqueurs)
    page_count = sum(1 for ln in lines if re.match(r"^\s*<!-- page:\d+ -->\s*$", ln))
    if page_count < 3:
        # Pas assez de pages pour la detection statistique
        # Fallback : texte identique >= 3 fois
        page_count = max(len(lines) // 30, 10)

    # Compter les occurrences de chaque ligne non-vide (normalisee)
    counter: Counter[str] = Counter()
    for ln in lines:
        stripped = ln.strip()
        if len(stripped) < min_length:
            continue
        if stripped.startswith("#") or stripped.startswith("<!-- page:"):
            continue
        # Normaliser les espaces pour la comparaison
        normalized = re.sub(r"\s+", " ", stripped)
        counter[normalized] += 1

    threshold = max(3, int(page_count * min_ratio))
    repeated = set()
    for text, count in counter.items():
        if count >= threshold:
            repeated.add(text)
            log.info(f"[clean] Header/footer detecte ({count}x) : "
                     f"{text[:80]}{'...' if len(text) > 80 else ''}")

    return repeated


def _remove_repeated_blocks(lines: list[str], repeated: set[str]) -> list[str]:
    """Supprime les lignes correspondant aux headers/footers detectes."""
    if not repeated:
        return lines
    cleaned = []
    for ln in lines:
        normalized = re.sub(r"\s+", " ", ln.strip())
        if normalized in repeated:
            continue
        cleaned.append(ln)
    return cleaned


# ---- Amelioration 2 : Nettoyage lignes parasites --------------------
def _clean_parasitic_lines(lines: list[str]) -> list[str]:
    """
    Supprime les lignes parasites :
    - Numeros de page isoles (ex: '15', 'Page 7')
    - Lignes <!-- image --> sans contexte
    - Lignes tres courtes qui ne sont ni titres, ni items de liste,
      ni cellules de tableau, ni marqueurs de page
    """
    cleaned = []
    for ln in lines:
        stripped = ln.strip()

        # Garder les lignes vides (structure), marqueurs de page, titres, listes, tableaux
        if not stripped:
            cleaned.append(ln)
            continue
        if stripped.startswith("<!-- page:"):
            cleaned.append(ln)
            continue
        if stripped.startswith("#"):
            cleaned.append(ln)
            continue
        if stripped.startswith("|") or stripped.startswith("-"):
            cleaned.append(ln)
            continue

        # Supprimer <!-- image --> (sera remplace par [Figure] si caption dispo)
        if stripped == "<!-- image -->":
            continue

        # Supprimer les numeros de page isoles : '15', 'Page 7', '- 15 -'
        if re.match(r"^[-–—]?\s*\d{1,4}\s*[-–—]?$", stripped):
            continue
        if re.match(r"^[Pp]age\s+\d+", stripped):
            continue

        # Supprimer les lignes de 1-2 caracteres non significatives
        if len(stripped) <= 2 and not stripped.startswith("-"):
            continue

        cleaned.append(ln)
    return cleaned


# ---- Amelioration 3 : Fusion paragraphes inter-pages -----------------
def _merge_broken_paragraphs(lines: list[str]) -> list[str]:
    """
    Fusionne les paragraphes coupes entre deux pages.
    Heuristique : si une ligne finit sans ponctuation finale
    et la suivante (en ignorant les marqueurs de page et lignes vides)
    commence par une minuscule ou une continuation, on fusionne.
    """
    if not lines:
        return lines

    result = []
    i = 0
    while i < len(lines):
        current = lines[i]
        stripped = current.strip()

        # Ne pas fusionner les titres, tableaux, listes, marqueurs
        if (not stripped or stripped.startswith("#") or stripped.startswith("|")
                or stripped.startswith("- ") or stripped.startswith("<!-- ")
                or stripped.startswith("[Figure")):
            result.append(current)
            i += 1
            continue

        # Chercher si cette ligne doit etre fusionnee avec la suivante
        # Condition : finit sans ponctuation terminale
        ends_without_punct = bool(
            stripped and stripped[-1] not in ".!?;:»\")]}–—"
            and not stripped.endswith("---")
            and len(stripped) > 20
        )

        if ends_without_punct:
            # Chercher la prochaine ligne de contenu (sauter marqueurs et vides)
            j = i + 1
            skipped = []
            while j < len(lines):
                next_stripped = lines[j].strip()
                if not next_stripped or next_stripped.startswith("<!-- page:"):
                    skipped.append(lines[j])
                    j += 1
                    continue
                break

            if j < len(lines):
                next_stripped = lines[j].strip()
                # La suite commence par une minuscule ou une continuation
                starts_continuation = bool(
                    next_stripped
                    and not next_stripped.startswith("#")
                    and not next_stripped.startswith("|")
                    and not next_stripped.startswith("- ")
                    and not next_stripped.startswith("[Figure")
                    and next_stripped[0].islower()
                )

                if starts_continuation:
                    # Fusionner : conserver les marqueurs de page entre les deux
                    for sk in skipped:
                        if sk.strip().startswith("<!-- page:"):
                            result.append(sk)
                    merged = stripped.rstrip() + " " + next_stripped.lstrip()
                    result.append(merged)
                    i = j + 1
                    continue

        result.append(current)
        i += 1

    return result


# ---- Amelioration 4 : Suppression sections boilerplate ---------------
# Patterns generiques de titres de sections a supprimer (TOC, TOF, etc.)
_BOILERPLATE_HEADING_PATTERNS = [
    r"table\s+of\s+content",
    r"table\s+of\s+figure",
    r"table\s+of\s+table",
    r"table\s+des\s+mati[eè]res",
    r"liste\s+des\s+figures",
    r"liste\s+des\s+tableaux",
    r"sommaire",
    r"table\s+des\s+illustrations",
]
_BOILERPLATE_RE = re.compile(
    r"^#{1,6}\s+(" + "|".join(_BOILERPLATE_HEADING_PATTERNS) + r")\s*$",
    re.IGNORECASE
)


def _remove_boilerplate_sections(lines: list[str]) -> list[str]:
    """
    Supprime les sections boilerplate (table des matieres, liste des figures, etc.)
    en detectant le heading et en supprimant tout jusqu'au prochain heading de meme
    niveau ou superieur.
    """
    result = []
    skip_until_depth = None  # profondeur du heading boilerplate a ignorer

    for ln in lines:
        stripped = ln.strip()

        # Detecter un heading
        heading_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading_match:
            depth = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            if skip_until_depth is not None:
                # On est en mode skip : arreter si on trouve un heading de meme niveau ou sup.
                if depth <= skip_until_depth:
                    skip_until_depth = None
                    # Ce heading est valide, on le garde
                else:
                    continue  # heading de sous-section du boilerplate, on skip

            # Verifier si ce heading est boilerplate
            if _BOILERPLATE_RE.match(stripped):
                log.info(f"[clean] Section boilerplate supprimee : {title}")
                skip_until_depth = depth
                continue

        elif skip_until_depth is not None:
            # On est en mode skip, ignorer le contenu
            continue

        result.append(ln)

    return result


# ---- Amelioration 5 : Validation tableaux Markdown -------------------
def _fix_markdown_tables(text: str) -> str:
    """
    Repare les tableaux Markdown courants :
    - Ajoute le separateur |---|---| manquant apres le header
    - Supprime les lignes de tableau repetees (headers de page re-exportes)
    """
    lines = text.split("\n")
    result = []
    seen_table_headers: dict[str, int] = {}  # header_normalized -> count
    i = 0

    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()

        if stripped.startswith("|") and stripped.endswith("|"):
            # Detecter les headers de tableau repetes
            normalized = re.sub(r"\s+", " ", stripped)
            seen_table_headers[normalized] = seen_table_headers.get(normalized, 0) + 1

            # Si c'est un header deja vu + suivi de son separateur, supprimer les deux
            if seen_table_headers[normalized] > 1:
                # Verifier si la ligne suivante est un separateur
                if i + 1 < len(lines) and re.match(r"^\s*\|[-|:\s]+\|\s*$", lines[i + 1]):
                    i += 2  # Sauter le header et son separateur
                    continue

            # Verifier qu'un separateur suit un header de tableau
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if (next_stripped.startswith("|") and next_stripped.endswith("|")
                        and not re.match(r"^\|[-|:\s]+\|$", next_stripped)):
                    # Pas de separateur apres le header, en ajouter un
                    col_count = stripped.count("|") - 1
                    if col_count > 0:
                        sep = "|" + "|".join(["---"] * col_count) + "|"
                        result.append(ln)
                        result.append(sep)
                        i += 1
                        continue

        result.append(ln)
        i += 1

    return "\n".join(result)


# ---- Amelioration 6 : Marquage references orphelines -----------------
def _mark_orphan_references(text: str) -> str:
    """
    Detecte les references a des figures/images dont le contenu n'est pas
    present dans le markdown et les marque clairement.
    Ex: 'Figure 3: TOE Boundary' seul sur une ligne -> '[Figure 3: TOE Boundary]'
    """
    lines = text.split("\n")
    result = []
    for ln in lines:
        stripped = ln.strip()
        # Ligne qui est uniquement une reference de figure sans contenu
        if re.match(r"^Figure\s+\d+\s*[:.]?\s*.{0,80}$", stripped, re.IGNORECASE):
            # Verifier que ce n'est pas dans un tableau ou un paragraphe
            if not stripped.startswith("|") and len(stripped) < 100:
                result.append(f"[{stripped}]")
                continue
        result.append(ln)
    return "\n".join(result)


# ═══════════════════════════════════════════════════════════════════════
# Fonction principale de nettoyage (combine les 6 ameliorations)
# ═══════════════════════════════════════════════════════════════════════
def clean_md(md_path: str, save_as: str = None) -> str:
    """
    Nettoyage avance d'un fichier Markdown produit par Docling.
    Applique 6 ameliorations dans l'ordre :

    1. Detection et suppression des headers/footers repetes (statistique)
    2. Nettoyage des lignes parasites (numeros de page, copyrights, etc.)
    3. Fusion des paragraphes coupes entre pages
    4. Suppression des sections boilerplate (TOC, TOF, sommaire)
    5. Validation et reparation des tableaux Markdown
    6. Marquage des references orphelines (figures sans image)
    """
    raw = Path(md_path).read_text(encoding="utf-8")
    if len(raw.strip()) < 10:
        log.warning(f"[clean_md] Fichier quasi vide : {md_path}")
        if save_as:
            Path(save_as).write_text("", encoding="utf-8")
        return ""

    lines = raw.split("\n")
    original_count = len(lines)
    log.info(f"[clean_md] Nettoyage de {Path(md_path).name} ({original_count} lignes)")

    # --- Amelioration 1 : Headers/footers repetes ---
    repeated = _detect_repeated_blocks(lines)
    lines = _remove_repeated_blocks(lines, repeated)
    after_hf = len(lines)

    # --- Amelioration 2 : Lignes parasites ---
    lines = _clean_parasitic_lines(lines)
    after_parasitic = len(lines)

    # --- Amelioration 4 : Sections boilerplate (avant fusion car la fusion
    #     ne doit pas merger a travers une section supprimee) ---
    lines = _remove_boilerplate_sections(lines)
    after_boilerplate = len(lines)

    # --- Amelioration 3 : Fusion paragraphes inter-pages ---
    lines = _merge_broken_paragraphs(lines)
    after_merge = len(lines)

    # Reassembler le texte
    txt = "\n".join(lines)

    # --- Amelioration 5 : Validation tableaux ---
    txt = _fix_markdown_tables(txt)

    # --- Amelioration 6 : References orphelines ---
    txt = _mark_orphan_references(txt)

    # Nettoyage final
    txt = re.sub(r"\n{3,}", "\n\n", txt)  # Reduire les sauts de ligne excessifs
    txt = re.sub(r"\*{3,}", "---", txt)   # Normaliser les separateurs
    txt = txt.strip() + "\n"

    # Rapport
    final_count = txt.count("\n") + 1
    log.info(f"[clean_md] Resultat : {original_count} -> {final_count} lignes")
    log.info(f"  Headers/footers supprimes : {original_count - after_hf}")
    log.info(f"  Lignes parasites          : {after_hf - after_parasitic}")
    log.info(f"  Sections boilerplate      : {after_parasitic - after_boilerplate}")
    log.info(f"  Paragraphes fusionnes     : {after_boilerplate - after_merge}")

    if save_as:
        Path(save_as).write_text(txt, encoding="utf-8")
        log.info(f"[clean_md] MD nettoye -> {save_as}")
    return txt


# ═══════════════════════════════════════════════════════════════════════
# Pipeline complet : PDF -> MD brut -> MD nettoye
# ═══════════════════════════════════════════════════════════════════════
def convert_and_clean(src_pdf: str, out_dir: str = "./out",
                      clean_dir: str = None) -> str:
    """
    Pipeline complet : convertit un PDF en Markdown puis applique
    le nettoyage avance.

    Args:
        src_pdf: chemin du fichier PDF source
        out_dir: dossier pour le Markdown brut
        clean_dir: dossier pour le Markdown nettoye (defaut: out_dir/../out_clean_md)

    Returns:
        Chemin du fichier Markdown nettoye
    """
    if clean_dir is None:
        clean_dir = str(Path(out_dir).parent / "out_clean_md")
    Path(clean_dir).mkdir(parents=True, exist_ok=True)

    # Etape 1 : conversion
    raw_md = pdf_to_md(src_pdf, out_dir=out_dir)

    # Etape 2 : nettoyage
    stem = Path(src_pdf).stem
    clean_path = str(Path(clean_dir) / f"{stem}-clean.md")
    clean_md(raw_md, save_as=clean_path)

    return clean_path


# Main
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "/home/marsattacks/Downloads/ANSSI-CC-cible_2011.pdf"

    clean_path = convert_and_clean(pdf_path, out_dir="./docs/out",
                                   clean_dir="./data/out_clean_md")
    print(f"\nResultat final : {clean_path}")

