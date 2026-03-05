import streamlit as st
import pandas as pd
import re
import json
import gspread
import unicodedata
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher

# ✅ MUST be before any other Streamlit UI calls
st.set_page_config(page_title="Find your Chogan Perfume", layout="wide")

# =========================================================
# CONFIG
# =========================================================

EXPECTED_EXTERNAL_COLS = [
    "Perfume",
    "Brand",
    "Gender",
    "Top Notes",
    "Heart Notes",
    "Base Notes",
    "All Notes",
    "Olfactory Family",
]

# Scoring blend
NOTE_WEIGHT = 0.70
VIBE_WEIGHT = 0.30

# ✅ Reduce inflation vs previous version
DNA_BOOST = 0.4          # was 0.7
ANCHOR_DIVISOR = 15.0    # was 10 (weaker anchor nudges)

MIN_SCORE_TO_SHOW = 3.0

# Threshold to show "Exact match" candidates for name search
EXACT_MATCH_SIM_THRESHOLD = 0.74  # tweakable (0.70–0.80)

# Precision/Recall blend for note scoring
# recall = "how many of my notes did you match?"
# precision = "how much of your perfume is actually relevant?"
NOTE_RECALL_WEIGHT = 0.70
NOTE_PRECISION_WEIGHT = 0.30


# =========================================================
# GOOGLE SHEETS
# =========================================================

@st.cache_resource
def get_gs_client():
    creds_info = json.loads(st.secrets["gcp_service_account"]["raw_json"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    return gspread.authorize(creds)


def get_external_worksheet():
    gc = get_gs_client()
    sheet_id = st.secrets["external_sheet"]["spreadsheet_id"]
    ws_name = st.secrets["external_sheet"]["worksheet_name"]
    return gc.open_by_key(sheet_id).worksheet(ws_name)


def ensure_external_headers(ws):
    headers = ws.row_values(1)
    if headers != EXPECTED_EXTERNAL_COLS:
        ws.clear()
        ws.append_row(EXPECTED_EXTERNAL_COLS)


def load_external_from_sheets():
    ws = get_external_worksheet()
    ensure_external_headers(ws)

    records = ws.get_all_records()
    df = pd.DataFrame(records)

    for col in EXPECTED_EXTERNAL_COLS:
        if col not in df.columns:
            df[col] = ""

    return df[EXPECTED_EXTERNAL_COLS], ws


def upsert_external_to_sheets(ws, row_dict):
    ws = get_external_worksheet()
    ensure_external_headers(ws)

    records = ws.get_all_records()

    key_perfume = str(row_dict.get("Perfume", "")).strip().lower()
    key_brand = str(row_dict.get("Brand", "")).strip().lower()

    target_row = None
    for i, r in enumerate(records, start=2):
        p = str(r.get("Perfume", "")).strip().lower()
        b = str(r.get("Brand", "")).strip().lower()
        if p == key_perfume and b == key_brand:
            target_row = i
            break

    values = [row_dict.get(c, "") for c in EXPECTED_EXTERNAL_COLS]

    if target_row:
        ws.update(f"A{target_row}:H{target_row}", [values])
    else:
        ws.append_row(values)


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c))


def norm_text(s: str) -> str:
    s = strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def name_similarity(a: str, b: str) -> float:
    a = norm_text(a)
    b = norm_text(b)

    fuzz = SequenceMatcher(None, a, b).ratio()
    ta = set(a.split())
    tb = set(b.split())
    jac = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0
    return (fuzz + jac) / 2.0


# =========================================================
# NOTES NORMALIZATION + “SMART” NOTE MATCHING
# =========================================================

NOTE_DESCRIPTORS = {
    "absolute", "essence", "accord", "note", "notes", "extract", "resinoid",
    "madagascan", "madagascar", "bourbon", "damask", "damascena",
    "african", "calabrian", "white", "black", "red", "pink", "green",
    "rich", "deep", "leathery", "soft", "warm", "fresh",
    # Keep these for pillars, but don’t let them erase the whole note:
    "woody", "woods",
}

NOTE_SYNONYMS = {
    "vanille": "vanilla",
    "bourbon vanilla": "vanilla",
    "bourbon vanilla absolute": "vanilla",
    "vanilla absolute": "vanilla",
    "vanilla orchid": "vanilla",

    "african orange blossom": "orange blossom",
    "orange blossom absolute": "orange blossom",
    "orange blossom essence": "orange blossom",

    "white musk": "musk",
    "cedarwood": "cedar",

    "oud wood": "oud",
    "oud wood accord": "oud",
    "oud accord": "oud",
    "agarwood": "oud",
}

def split_notes(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    parts = re.split(r"[,/;]+", str(x))
    return [p.strip() for p in parts if p.strip()]

def normalize_note(note: str) -> str:
    raw = strip_accents(str(note)).lower()
    raw = re.sub(r"[^a-z0-9\s]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        return ""

    if raw in NOTE_SYNONYMS:
        return NOTE_SYNONYMS[raw]

    tokens = [t for t in raw.split() if t and t not in NOTE_DESCRIPTORS]
    if not tokens:
        tokens = raw.split()

    cleaned = " ".join(tokens).strip()

    if cleaned in NOTE_SYNONYMS:
        return NOTE_SYNONYMS[cleaned]

    return cleaned

def normalize_notes_list(lst):
    out = []
    for x in lst:
        n = normalize_note(x)
        if n:
            out.append(n)
    return out

def notes_match(a: str, b: str) -> bool:
    a = normalize_note(a)
    b = normalize_note(b)
    if not a or not b:
        return False
    if a == b:
        return True
    return (a in b) or (b in a)

def set_intersection_smart(query_notes_set: set[str], perfume_notes_set: set[str]) -> set[str]:
    matched = set()
    for q in query_notes_set:
        for p in perfume_notes_set:
            if notes_match(q, p):
                matched.add(q)
                break
    return matched

def set_missing_smart(query_notes_set: set[str], perfume_notes_set: set[str]) -> set[str]:
    missing = set()
    for q in query_notes_set:
        ok = False
        for p in perfume_notes_set:
            if notes_match(q, p):
                ok = True
                break
        if not ok:
            missing.add(q)
    return missing


# =========================================================
# PILLARS (VIBE / ACCORDS)
# =========================================================

# ✅ Citrussy accord created + made less “everything matches”
# (We keep more distinctive citrus signals, not ultra-common "orange/lemon")
PILLARS = {
    "fruity": {
        "pear", "raspberry", "strawberry", "lychee", "blackcurrant", "currant",
        "peach", "plum", "apple", "mango",
    },
    "citrussy": {
        "bergamot", "grapefruit", "lime", "yuzu", "citron", "pomelo",
        "blood orange", "orange zest", "lemon zest",
        "neroli", "petitgrain",
        "citrus",
    },
    "floral": {
        "rose", "black rose", "jasmine", "tuberose", "orange blossom", "peony",
        "datura", "iris", "violet", "orchid",
    },
    "gourmand": {
        "vanilla", "praline", "caramel", "coffee", "tonka", "chocolate",
        "benzoin", "toffee", "cocoa", "honey",
    },
    "woody": {
        "patchouli", "cedar", "cedarwood", "sandalwood", "vetiver",
    "moss", "oakmoss", "papyrus", "oud",

    # ✅ modern woods / woody accords (needed for By the Fireplace etc.)
    "guaiac", "guaiac wood",
    "cashmeran",
    "iso e super", "iso e",   # optional but useful if you ever store it
    "amberwood",
    "dry wood", "woody accord", "wood accord",
    "smoke", "smoky",         # optional, often used with woods
    "juniper",                # woody-aromatic
    },
    "musky": {"musk", "ambroxan", "ambergris"},
    "resinous": {"incense", "labdanum", "amber", "myrrh", "opoponax"},
}

ANCHOR_COMBOS = [
    ({"rose", "patchouli"}, 0.8),
    ({"vanilla", "patchouli"}, 0.6),
    ({"coffee", "vanilla"}, 0.8),
    ({"praline", "vanilla"}, 0.7),
]

def detect_pillars(notes_set: set[str]) -> set[str]:
    found = set()
    notes_blob = " ".join(sorted(notes_set))
    for pillar, kws in PILLARS.items():
        for kw in kws:
            if kw in notes_set or kw in notes_blob:
                found.add(pillar)
                break
    return found
    
def penalties(query_pillars: set[str], perfume_pillars: set[str]) -> float:
    pen = 0.0
    if "gourmand" in query_pillars and "gourmand" not in perfume_pillars:
        pen -= 1.0
    return pen


# =========================================================
# SCORING (PYRAMID + VIBE) WITH PRECISION+RECALL
# =========================================================

def get_row_note_sets(row):
    top = set(normalize_notes_list(split_notes(row.get("Top Notes", ""))))
    heart = set(normalize_notes_list(split_notes(row.get("Heart Notes", ""))))
    base = set(normalize_notes_list(split_notes(row.get("Base Notes", ""))))
    all_notes = top | heart | base
    return top, heart, base, all_notes

def pyramid_weights_for_sets(top: set[str], heart: set[str], base: set[str]) -> float:
    # same weights as your pyramid scoring
    return (2.0 * len(top)) + (1.6 * len(heart)) + (1.4 * len(base))

def score_notes_pyramid(query_top, query_heart, query_base, perf_top, perf_heart, perf_base):
    """
    Pyramid-aware overlap using smart note matching.
    Returns:
      recall-like score in [0..1] + weighted_match for precision computation.
    """
    q_top = set(query_top)
    q_heart = set(query_heart)
    q_base = set(query_base)

    denom = (2.0 * len(q_top)) + (1.6 * len(q_heart)) + (1.4 * len(q_base))
    if denom <= 0:
        return 0.0, 0.0

    s = 0.0

    for n in q_top:
        if any(notes_match(n, p) for p in perf_top):
            s += 2.0
        elif any(notes_match(n, p) for p in perf_heart):
            s += 1.2
        elif any(notes_match(n, p) for p in perf_base):
            s += 0.8

    for n in q_heart:
        if any(notes_match(n, p) for p in perf_heart):
            s += 1.6
        elif any(notes_match(n, p) for p in perf_top):
            s += 1.2
        elif any(notes_match(n, p) for p in perf_base):
            s += 1.0

    for n in q_base:
        if any(notes_match(n, p) for p in perf_base):
            s += 1.4
        elif any(notes_match(n, p) for p in perf_heart):
            s += 1.1

    recall_like = max(min(s / denom, 1.0), 0.0)
    return recall_like, s

def score_notes_simple(query_notes: set[str], perfume_notes: set[str]) -> tuple[float, int]:
    """
    Returns (recall_like, matched_count)
    """
    if not query_notes:
        return 0.0, 0
    matched = set_intersection_smart(query_notes, perfume_notes)
    recall_like = len(matched) / max(len(query_notes), 1)
    return recall_like, len(matched)

def score_perfume(query_notes, row, used_pyramid=False, query_top=None, query_heart=None, query_base=None):
    perf_top, perf_heart, perf_base, perf_all = get_row_note_sets(row)

    # NOTE SCORE (0..1) using both recall + precision
    if used_pyramid and (query_top or query_heart or query_base):
        recall_like, weighted_match = score_notes_pyramid(
            query_top, query_heart, query_base, perf_top, perf_heart, perf_base
        )

        perf_weight_total = pyramid_weights_for_sets(perf_top, perf_heart, perf_base)
        precision_like = (weighted_match / perf_weight_total) if perf_weight_total > 0 else 0.0

        note_score = (NOTE_RECALL_WEIGHT * recall_like) + (NOTE_PRECISION_WEIGHT * precision_like)

        q_all_for_vibe = set(query_top) | set(query_heart) | set(query_base)
        query_all_for_missing = set(query_top) | set(query_heart) | set(query_base)
    else:
        recall_like, matched_count = score_notes_simple(query_notes, perf_all)
        precision_like = matched_count / max(len(perf_all), 1) if perf_all else 0.0

        note_score = (NOTE_RECALL_WEIGHT * recall_like) + (NOTE_PRECISION_WEIGHT * precision_like)

        q_all_for_vibe = set(query_notes)
        query_all_for_missing = set(query_notes)

    note_score = max(min(note_score, 1.0), 0.0)

    # VIBE SCORE (0..1)
    query_pillars = detect_pillars(q_all_for_vibe)
    perfume_pillars = detect_pillars(perf_all)

    pillar_overlap = len(query_pillars & perfume_pillars)
    vibe_score = pillar_overlap / max(len(query_pillars), 1) if query_pillars else 0.0

    blended = NOTE_WEIGHT * note_score + VIBE_WEIGHT * vibe_score

    # Anchors (weaker than before)
    for combo, b in ANCHOR_COMBOS:
        if combo <= q_all_for_vibe and combo <= perf_all:
            blended += b / ANCHOR_DIVISOR

    blended += penalties(query_pillars, perfume_pillars) / 10.0

    # DNA boost if vibe overlap strong (reduced)
    if pillar_overlap >= 3:
        blended += DNA_BOOST / 10.0

    score10 = max(min(blended * 10.0, 10.0), 0.0)

    matched_notes = sorted(set_intersection_smart(query_all_for_missing, perf_all))
    missing_notes = sorted(set_missing_smart(query_all_for_missing, perf_all))
    matched_pillars = sorted(query_pillars & perfume_pillars)

    return score10, matched_notes, matched_pillars, missing_notes


# =========================================================
# DATA LOAD
# =========================================================

@st.cache_data
def load_chogan_csv(path):
    return pd.read_csv(path)

try:
    chogan = load_chogan_csv("chogan_catalog.csv")
except Exception:
    st.error("Could not load chogan_catalog.csv. Make sure it is in your repo.")
    st.stop()

try:
    external, external_ws = load_external_from_sheets()
except Exception as e:
    st.error(f"Could not load external perfumes from Google Sheets: {e}")
    external = pd.DataFrame(columns=EXPECTED_EXTERNAL_COLS)
    external_ws = None


# =========================================================
# UI HELPERS
# =========================================================

def score_badge(score: float):
    if score >= 7.0:
        return ("Excellent match", "#16a34a", "white")  # green
    if score >= 5.0:
        return ("Good match", "#60a5fa", "white")       # light blue
    if score >= 3.0:
        return ("Worth a try", "#fde68a", "#111827")    # light yellow
    return ("Low match", "#e5e7eb", "#111827")          # gray

def score_scale_card():
    st.markdown(
        """
<div style="border:1px solid #e5e7eb; border-radius:12px; padding:12px; background:#fff;">
  <div style="font-weight:600; margin-bottom:8px;">How to read the match score</div>
  <div style="display:flex; gap:8px; flex-wrap:wrap;">
    <span style="background:#16a34a; color:#fff; padding:6px 10px; border-radius:999px; font-weight:600; font-size:0.9rem;">7.0–10.0 — Excellent match</span>
    <span style="background:#60a5fa; color:#fff; padding:6px 10px; border-radius:999px; font-weight:600; font-size:0.9rem;">5.0–6.9 — Good match</span>
    <span style="background:#fde68a; color:#111827; padding:6px 10px; border-radius:999px; font-weight:600; font-size:0.9rem;">3.0–4.9 — Worth a try</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# SESSION STATE
# =========================================================

if "has_searched" not in st.session_state:
    st.session_state.has_searched = False

if "last_query" not in st.session_state:
    st.session_state.last_query = {}

def reset_search():
    st.session_state.has_searched = False
    st.session_state.last_query = {}


# =========================================================
# APP
# =========================================================

st.title("Find your Chogan Perfume")

tab_search, tab_add = st.tabs(["Search", "Add a new perfume to the database"])

# =========================================================
# TAB 1: SEARCH
# =========================================================

with tab_search:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Search")

        mode = st.radio("Choose input type:", ["By perfume name", "By notes only"])

        perfume_name = ""
        brand_name = ""

        if mode == "By perfume name":
            perfume_name = st.text_input("Perfume name (e.g., Nina)")
            brand_name = st.text_input("Brand (optional)")
            notes_text = st.text_input("Desired notes (comma-separated)")
        else:
            notes_text = st.text_input("Desired notes (comma-separated)", placeholder="e.g., jasmine, vanilla, patchouli")

        gender_choice = st.selectbox(
            "Gender preference (optional)",
            [
                "Any",
                "Women (F)",
                "Men (M)",
                "Unisex (U)",
                "Women or Unisex (F/U)",
                "Men or Unisex (M/U)",
            ],
        )

        top_n = st.slider("How many recommendations?", 1, 5, 3)

        cbtn1, cbtn2 = st.columns([1, 1])
        with cbtn1:
            search_clicked = st.button("Search", use_container_width=True)
        with cbtn2:
            st.button("Reset", on_click=reset_search, use_container_width=True)

        if search_clicked:
            st.session_state.has_searched = True
            st.session_state.last_query = {
                "mode": mode,
                "perfume_name": perfume_name,
                "brand_name": brand_name,
                "notes_text": notes_text,
                "gender_choice": gender_choice,
                "top_n": top_n,
            }

    with right:
        st.subheader("My Recommendations")

        # Score scale card always visible on desktop right column
        score_scale_card()

        if not st.session_state.has_searched:
            st.info("Enter your search, then click **Search**.")
        else:
            qd = st.session_state.last_query
            mode = qd.get("mode", "By perfume name")
            perfume_name = qd.get("perfume_name", "")
            brand_name = qd.get("brand_name", "")
            notes_text = qd.get("notes_text", "")
            gender_choice = qd.get("gender_choice", "Any")
            top_n = int(qd.get("top_n", 3))

            # 1) Build query notes
            raw_notes = normalize_notes_list(split_notes(notes_text))
            query_notes = set(raw_notes)

            used_pyramid = False
            query_top, query_heart, query_base = set(), set(), set()

            # 2) Exact matches by similarity (shows BOTH Good Girl etc.)
            exact_hits = chogan.iloc[0:0]
            if mode == "By perfume name" and perfume_name.strip():
                qname = perfume_name.strip()
                hit_rows = []
                q_norm = norm_text(qname)

                for idx, row in chogan.iterrows():
                    insp = str(row.get("Inspiration", ""))
                    if not insp:
                        continue
                    insp_norm = norm_text(insp)

                    sim = name_similarity(qname, insp)

                    if brand_name.strip():
                        b = norm_text(brand_name)
                        if b and b in insp_norm:
                            sim += 0.06

                    # Token containment: avoids “Good Girl” being hijacked by a longer string
                    contains = q_norm in insp_norm

                    if sim >= EXACT_MATCH_SIM_THRESHOLD or contains:
                        hit_rows.append((sim, idx))

                hit_rows.sort(key=lambda x: x[0], reverse=True)
                exact_idxs = [idx for _, idx in hit_rows[:10]]
                if exact_idxs:
                    exact_hits = chogan.loc[exact_idxs].copy()

            if len(exact_hits) > 0:
                st.success(f"Exact match found in Chogan inspirations ({len(exact_hits)} result(s)).")
                for _, hit in exact_hits.head(top_n).iterrows():
                    ref = (
                        hit.get("Perfume reference")
                        or hit.get("Perfume ref.")
                        or hit.get("Reference")
                        or hit.get("Code")
                        or hit.get("ID")
                        or ""
                    )
                    st.markdown(f"### ✅ Exact match — **{ref}**")
                    st.write(f"Inspiration: *{hit.get('Inspiration','')}*")
                    st.write(f"Top: {hit.get('Top Notes','')}")
                    st.write(f"Heart: {hit.get('Heart Notes','')}")
                    st.write(f"Base: {hit.get('Base Notes','')}")
                    st.divider()

            # 3) External DB notes
            if mode == "By perfume name" and perfume_name.strip() and not external.empty:
                mask = external["Perfume"].fillna("").astype(str).str.lower().str.contains(perfume_name.strip().lower(), na=False)
                matches = external[mask]

                if brand_name.strip() and len(matches) > 1:
                    bmask = matches["Brand"].fillna("").astype(str).str.lower().str.contains(brand_name.strip().lower(), na=False)
                    matches = matches[bmask]

                if len(matches) > 0:
                    used_external = matches.iloc[0].to_dict()

                    etop = set(normalize_notes_list(split_notes(used_external.get("Top Notes", ""))))
                    ehe = set(normalize_notes_list(split_notes(used_external.get("Heart Notes", ""))))
                    eba = set(normalize_notes_list(split_notes(used_external.get("Base Notes", ""))))

                    if etop or ehe or eba:
                        query_top, query_heart, query_base = etop, ehe, eba
                        used_pyramid = True
                        query_notes |= (etop | ehe | eba)
                    else:
                        eall = set(normalize_notes_list(split_notes(used_external.get("All Notes", ""))))
                        query_notes |= eall
                        used_pyramid = False

                    st.info(f"Using saved notes for: {used_external.get('Perfume','')} ({used_external.get('Brand','')})")

            # 4) If still no notes, seed from first exact match
            if (not query_notes) and len(exact_hits) > 0:
                seed = exact_hits.iloc[0].to_dict()
                st.info("Using the exact match notes to generate recommendations.")

                qtop = set(normalize_notes_list(split_notes(seed.get("Top Notes", ""))))
                qhe = set(normalize_notes_list(split_notes(seed.get("Heart Notes", ""))))
                qba = set(normalize_notes_list(split_notes(seed.get("Base Notes", ""))))

                if qtop or qhe or qba:
                    query_top, query_heart, query_base = qtop, qhe, qba
                    used_pyramid = True
                    query_notes |= (qtop | qhe | qba)
                else:
                    query_notes |= set(normalize_notes_list(split_notes(seed.get("All Notes", ""))))
                    used_pyramid = False

            # 5) Gender filter
            filtered = chogan.copy()
            if "Gender" in filtered.columns:
                g_raw = filtered["Gender"].fillna("").astype(str).str.strip().str.upper()

                def norm_gender(val: str) -> str:
                    v = val.strip().upper()
                    if v in {"F", "M", "U"}:
                        return v
                    if "UNISEX" in v:
                        return "U"
                    if "WOM" in v or "FEM" in v:
                        return "F"
                    if "MEN" in v or "MASC" in v:
                        return "M"
                    if "F" in v and "U" in v:
                        return "F/U"
                    if "M" in v and "U" in v:
                        return "M/U"
                    return v

                g = g_raw.apply(norm_gender)

                if gender_choice == "Women (F)":
                    filtered = filtered[g.isin(["F", "F/U"])]
                elif gender_choice == "Men (M)":
                    filtered = filtered[g.isin(["M", "M/U"])]
                elif gender_choice == "Unisex (U)":
                    filtered = filtered[g.isin(["U", "F/U", "M/U"])]
                elif gender_choice == "Women or Unisex (F/U)":
                    filtered = filtered[g.isin(["F", "U", "F/U"])]
                elif gender_choice == "Men or Unisex (M/U)":
                    filtered = filtered[g.isin(["M", "U", "M/U"])]
                # Any -> no filter

            # 6) Recommendations
            if not query_notes and not used_pyramid:
                st.warning("Add some notes, or search a perfume name that exists in your external database.")
            else:
                exact_refs = set()
                if len(exact_hits) > 0:
                    for _, hit in exact_hits.iterrows():
                        ref = (
                            hit.get("Perfume reference")
                            or hit.get("Perfume ref.")
                            or hit.get("Reference")
                            or hit.get("Code")
                            or hit.get("ID")
                            or ""
                        )
                        exact_refs.add(str(ref).strip().lower())

                results = []
                for _, row in filtered.iterrows():
                    ref = (
                        row.get("Perfume reference")
                        or row.get("Perfume ref.")
                        or row.get("Reference")
                        or row.get("Code")
                        or row.get("ID")
                        or ""
                    )
                    if str(ref).strip().lower() in exact_refs:
                        continue

                    score, matched_notes, matched_pillars, missing_notes = score_perfume(
                        query_notes=query_notes,
                        row=row,
                        used_pyramid=used_pyramid,
                        query_top=query_top,
                        query_heart=query_heart,
                        query_base=query_base,
                    )

                    if mode == "By perfume name" and perfume_name.strip():
                        sim = name_similarity(perfume_name, str(row.get("Inspiration", "")))
                        if sim > 0.85:
                            score = min(score + 1.0, 10.0)
                        elif sim > 0.70:
                            score = min(score + 0.5, 10.0)

                    results.append((score, matched_notes, matched_pillars, missing_notes, row))

                results.sort(key=lambda x: x[0], reverse=True)

                shown = 0
                for score, matched_notes, matched_pillars, missing_notes, row in results:
                    if score < MIN_SCORE_TO_SHOW:
                        continue

                    shown += 1
                    ref = (
                        row.get("Perfume reference")
                        or row.get("Perfume ref.")
                        or row.get("Reference")
                        or row.get("Code")
                        or row.get("ID")
                        or ""
                    )

                    st.markdown(f"### #{shown} — **{ref}**")
                    st.write(f"Inspiration: *{row.get('Inspiration','')}*")

                    label, bg, fg = score_badge(score)
                    st.markdown(
                        f"""
<div style="border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px; background:#fff; display:flex; align-items:center; justify-content:space-between;">
  <div style="font-size:1.25rem; font-weight:900;">Match score: {score:.2f} / 10</div>
  <div style="background:{bg}; color:{fg}; padding:6px 12px; border-radius:999px; font-weight:700; font-size:0.9rem;">
    {label}
  </div>
</div>
                        """,
                        unsafe_allow_html=True,
                    )

                    notes_txt = ", ".join(matched_notes) if matched_notes else "None"
                    accords_txt = ", ".join(matched_pillars) if matched_pillars else "None"
                    st.write(f"**Matched notes:** {notes_txt}  |  **Matched accords:** {accords_txt}")

                    st.write(f"Top: {row.get('Top Notes','')}")
                    st.write(f"Heart: {row.get('Heart Notes','')}")
                    st.write(f"Base: {row.get('Base Notes','')}")

                    if missing_notes:
                        miss_txt = ", ".join(missing_notes)
                        st.markdown(
                            f"<div style='color:#9ca3af; font-style:italic; margin-top:6px;'>Missing notes: {miss_txt}</div>",
                            unsafe_allow_html=True,
                        )

                    st.divider()

                    if shown >= top_n:
                        break

                if shown == 0:
                    st.warning(
                        "Sorry, we don't have a good match for the notes in the perfume you are looking for. "
                        "Would you like to try something else?"
                    )


# =========================================================
# TAB 2: ADD EXTERNAL PERFUME
# =========================================================

with tab_add:
    st.subheader("Add a new perfume to the database")

    with st.form("add_external", clear_on_submit=True):
        c1, c2 = st.columns(2)

        with c1:
            new_perfume = st.text_input("Perfume")
            new_brand = st.text_input("Brand")
            new_gender = st.selectbox("Gender", ["", "F", "M", "U"])
        with c2:
            new_top = st.text_input("Top Notes (comma-separated)")
            new_heart = st.text_input("Heart Notes (comma-separated)")
            new_base = st.text_input("Base Notes (comma-separated)")
            new_all = st.text_input("All Notes (comma-separated) — use if no pyramid")

        submitted = st.form_submit_button("Save external perfume")

    if submitted:
        if not new_perfume.strip():
            st.error("Perfume name required.")
        elif external_ws is None:
            st.error("Google Sheets not connected. Fix the error above first.")
        else:
            row_dict = {
                "Perfume": new_perfume.strip(),
                "Brand": new_brand.strip(),
                "Gender": new_gender.strip(),
                "Top Notes": new_top.strip(),
                "Heart Notes": new_heart.strip(),
                "Base Notes": new_base.strip(),
                "All Notes": new_all.strip(),
                "Olfactory Family": "",
            }
            upsert_external_to_sheets(external_ws, row_dict)
            st.success("Saved.")
            st.rerun()

    try:
        external_latest, _ = load_external_from_sheets()
    except Exception:
        external_latest = external

    with st.expander("View saved external perfumes"):
        st.dataframe(external_latest.tail(50))
