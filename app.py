import streamlit as st
import pandas as pd
import re
import json
import gspread
import unicodedata
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher

st.set_page_config(page_title="Find your Chogan Perfume", layout="centered")

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

NOTE_WEIGHT = 0.72
VIBE_WEIGHT = 0.28
DNA_BOOST = 0.35
MIN_SCORE_TO_SHOW = 3.0
EXACT_MATCH_SIM_THRESHOLD = 0.74
F_BETA = 0.85

# =========================================================
# SESSION STATE
# =========================================================
if "view" not in st.session_state:
    st.session_state.view = "search"

if "last_query" not in st.session_state:
    st.session_state.last_query = {}


def reset_search():
    st.session_state.last_query = {}
    st.session_state.view = "search"


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


@st.cache_data(ttl=300)
def load_external_from_sheets_cached():
    ws = get_external_worksheet()
    ensure_external_headers(ws)
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    for col in EXPECTED_EXTERNAL_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[EXPECTED_EXTERNAL_COLS]


def upsert_external_to_sheets(row_dict):
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

    load_external_from_sheets_cached.clear()


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
# NOTE MATCHING
# =========================================================
NOTE_DESCRIPTORS = {
    "absolute", "essence", "accord", "note", "notes", "extract", "resinoid",
    "madagascan", "madagascar", "bourbon", "damask", "damascena", "honeyed",
    "african", "calabrian", "white", "black", "red", "pink", "green",
    "rich", "deep", "leathery", "soft", "warm", "fresh",
    "wood", "woody", "woods", "noire", "noir", "intense",
    "eau", "de", "parfum", "edp",
}

NOTE_SYNONYMS = {
    "vanille": "vanilla",
    "bourbon vanilla": "vanilla",
    "bourbon vanilla absolute": "vanilla",
    "vanilla absolute": "vanilla",
    "vanilla orchid": "vanilla",
    "african orange blossom": "orange blossom",
    "orange blossom": "orange blossom",
    "orange flower": "orange blossom",
    "oud wood": "oud",
    "oud wood accord": "oud",
    "oud accord": "oud",
    "agarwood": "oud",
    "white musk": "musk",
    "cedarwood": "cedar",
    "amberwood": "amber wood",
    "black currant": "blackcurrant",
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
    return a == b or a in b or b in a


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
        if not any(notes_match(q, p) for p in perfume_notes_set):
            missing.add(q)
    return missing


def set_extra_smart(query_notes_set: set[str], perfume_notes_set: set[str]) -> set[str]:
    extra = set()
    for p in perfume_notes_set:
        if not any(notes_match(q, p) for q in query_notes_set):
            extra.add(p)
    return extra


# =========================================================
# PILLARS
# =========================================================
PILLARS = {
    "fruity": {
        "pear", "raspberry", "strawberry", "lychee", "blackcurrant", "currant",
        "peach", "plum", "apple", "mango", "pineapple", "cassis",
    },
    "citrussy": {
        "bergamot", "lemon", "lime", "orange", "mandarin", "tangerine",
        "grapefruit", "pomelo", "yuzu", "citron", "blood orange",
        "orange zest", "lemon zest", "citrus", "citric", "neroli", "petitgrain",
    },
    "floral": {
        "rose", "black rose", "jasmine", "tuberose", "orange blossom", "peony",
        "datura", "iris", "violet", "orchid", "lily of the valley",
    },
    "gourmand": {
        "vanilla", "praline", "caramel", "coffee", "tonka", "chocolate",
        "benzoin", "toffee", "cocoa", "honey", "sugar", "marshmallow",
    },
    "woody": {
        "patchouli", "cedar", "sandalwood", "vetiver", "moss", "oakmoss",
        "papyrus", "oud", "cashmeran", "guaiac", "guaiac wood",
        "amber wood", "amberwood", "cedarwood", "dry woods", "woods",
    },
    "musky": {"musk", "ambroxan", "ambergris", "ambrox"},
    "resinous": {"incense", "labdanum", "amber", "myrrh", "opoponax", "resin"},
}

ANCHOR_COMBOS = [
    ({"rose", "patchouli"}, 0.8),
    ({"vanilla", "patchouli"}, 0.6),
    ({"coffee", "vanilla"}, 0.8),
    ({"praline", "vanilla"}, 0.7),
    ({"neroli", "orange blossom"}, 0.5),
]


def detect_pillars(notes_set: set[str]) -> set[str]:
    found = set()
    for pillar, kws in PILLARS.items():
        if notes_set & kws:
            found.add(pillar)
    return found


def penalties(query_pillars: set[str], perfume_pillars: set[str]) -> float:
    pen = 0.0
    if "gourmand" in query_pillars and "gourmand" not in perfume_pillars:
        pen -= 1.0
    return pen


# =========================================================
# SCORING
# =========================================================
def get_row_note_sets(row):
    top = set(normalize_notes_list(split_notes(row.get("Top Notes", ""))))
    heart = set(normalize_notes_list(split_notes(row.get("Heart Notes", ""))))
    base = set(normalize_notes_list(split_notes(row.get("Base Notes", ""))))
    all_notes = top | heart | base
    return top, heart, base, all_notes


def fbeta(precision: float, recall: float, beta: float) -> float:
    if precision <= 0 and recall <= 0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision) + recall
    if denom <= 0:
        return 0.0
    return (1 + b2) * (precision * recall) / denom


def score_notes_simple(query_notes: set[str], perfume_notes: set[str]):
    if not query_notes:
        return 0.0, set(), set(), set()
    matched = set_intersection_smart(query_notes, perfume_notes)
    missing = set_missing_smart(query_notes, perfume_notes)
    extra = set_extra_smart(query_notes, perfume_notes)
    recall = len(matched) / max(len(query_notes), 1)
    precision = len(matched) / max(len(perfume_notes), 1)
    note_score = fbeta(precision, recall, F_BETA)
    return note_score, matched, missing, extra


def score_notes_pyramid(query_top, query_heart, query_base, perf_top, perf_heart, perf_base):
    q_top = set(query_top)
    q_heart = set(query_heart)
    q_base = set(query_base)

    denom_q = (2.0 * len(q_top)) + (1.6 * len(q_heart)) + (1.4 * len(q_base))
    denom_p = (2.0 * len(perf_top)) + (1.6 * len(perf_heart)) + (1.4 * len(perf_base))

    if denom_q <= 0:
        return 0.0, set(), set(), set()

    matched_weight = 0.0
    matched_notes = set()

    def any_match(n: str, s: set[str]) -> bool:
        return any(notes_match(n, p) for p in s)

    for n in q_top:
        if any_match(n, perf_top):
            matched_weight += 2.0
            matched_notes.add(n)
        elif any_match(n, perf_heart):
            matched_weight += 1.2
            matched_notes.add(n)
        elif any_match(n, perf_base):
            matched_weight += 0.8
            matched_notes.add(n)

    for n in q_heart:
        if any_match(n, perf_heart):
            matched_weight += 1.6
            matched_notes.add(n)
        elif any_match(n, perf_top):
            matched_weight += 1.2
            matched_notes.add(n)
        elif any_match(n, perf_base):
            matched_weight += 1.0
            matched_notes.add(n)

    for n in q_base:
        if any_match(n, perf_base):
            matched_weight += 1.4
            matched_notes.add(n)
        elif any_match(n, perf_heart):
            matched_weight += 1.1
            matched_notes.add(n)

    recall = matched_weight / max(denom_q, 1e-9)
    precision = matched_weight / max(denom_p, 1e-9) if denom_p > 0 else 0.0
    note_score = fbeta(precision, recall, F_BETA)

    query_all = q_top | q_heart | q_base
    perf_all = perf_top | perf_heart | perf_base
    missing = set_missing_smart(query_all, perf_all)
    extra = set_extra_smart(query_all, perf_all)

    return max(min(note_score, 1.0), 0.0), matched_notes, missing, extra


def score_perfume(query_notes, row, used_pyramid=False, query_top=None, query_heart=None, query_base=None):
    perf_top, perf_heart, perf_base, perf_all = get_row_note_sets(row)

    query_top = query_top or set()
    query_heart = query_heart or set()
    query_base = query_base or set()

    if used_pyramid and (query_top or query_heart or query_base):
        note_score, matched_qnotes, missing_qnotes, extra_pnotes = score_notes_pyramid(
            query_top, query_heart, query_base, perf_top, perf_heart, perf_base
        )
        q_all_for_vibe = set(query_top) | set(query_heart) | set(query_base)
    else:
        note_score, matched_qnotes, missing_qnotes, extra_pnotes = score_notes_simple(query_notes, perf_all)
        q_all_for_vibe = set(query_notes)

    query_pillars = detect_pillars(q_all_for_vibe)
    perfume_pillars = detect_pillars(perf_all)

    pillar_overlap = len(query_pillars & perfume_pillars)
    vibe_score = pillar_overlap / max(len(query_pillars), 1) if query_pillars else 0.0

    blended = NOTE_WEIGHT * note_score + VIBE_WEIGHT * vibe_score

    for combo, b in ANCHOR_COMBOS:
        if combo <= q_all_for_vibe and combo <= perf_all:
            blended += b / 10.0

    blended += penalties(query_pillars, perfume_pillars) / 10.0

    if pillar_overlap >= 3:
        blended += DNA_BOOST / 10.0

    score10 = max(min(blended * 10.0, 10.0), 0.0)

    matched_notes = sorted(matched_qnotes)
    missing_notes = sorted(missing_qnotes)
    extra_notes = sorted(list(extra_pnotes))[:10]
    matched_pillars = sorted(query_pillars & perfume_pillars)

    return score10, matched_notes, matched_pillars, missing_notes, extra_notes


# =========================================================
# DATA LOAD
# =========================================================
@st.cache_data
def load_chogan_csv(path):
    return pd.read_csv(path)


try:
    chogan = load_chogan_csv("chogan_catalog.csv")
except Exception as e:
    st.error(f"Could not load chogan_catalog.csv: {e}")
    st.stop()

external = pd.DataFrame(columns=EXPECTED_EXTERNAL_COLS)
try:
    external = load_external_from_sheets_cached()
except Exception as e:
    st.warning(f"Could not load external perfumes from Google Sheets: {e}")


# =========================================================
# UI HELPERS
# =========================================================
def score_badge(score: float):
    if score >= 7.0:
        return ("Excellent match", "#16a34a", "white")
    if score >= 5.0:
        return ("Good match", "#60a5fa", "white")
    if score >= 3.0:
        return ("Worth a try", "#fde68a", "#111827")
    return ("Low match", "#e5e7eb", "#111827")


def score_scale_card():
    st.markdown(
        """
<div class="score-scale-card">
  <div class="score-scale-title">How to read the match score</div>
  <div class="score-scale-row">
    <span class="pill pill-green">7.0–10.0 — Excellent match</span>
    <span class="pill pill-blue">5.0–6.9 — Good match</span>
    <span class="pill pill-yellow">3.0–4.9 — Worth a try</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def compact_summary(matched_notes, matched_pillars, missing_notes):
    return (
        f"<div class='summary-line'>"
        f"<b>Matched notes:</b> {len(matched_notes)}"
        f" &nbsp;|&nbsp; "
        f"<b>Accords:</b> {', '.join(matched_pillars) if matched_pillars else 'None'}"
        f" &nbsp;|&nbsp; "
        f"<b>Missing:</b> {len(missing_notes)}"
        f"</div>"
    )


def why_this_match_text(score, matched_notes, matched_pillars, missing_notes):
    parts = []

    if matched_notes:
        parts.append(f"It matches on key notes such as {', '.join(matched_notes[:5])}.")
    else:
        parts.append("It is driven more by overall accord similarity than direct note overlap.")

    if matched_pillars:
        parts.append(f"The shared vibe comes mainly from {', '.join(matched_pillars)} accords.")

    if missing_notes:
        parts.append(f"It is missing {len(missing_notes)} of your query notes, so treat it as a stylistic match rather than an exact clone.")

    if score >= 7:
        parts.append("This is a very close fit overall.")
    elif score >= 5:
        parts.append("This is a solid match with a few trade-offs.")
    else:
        parts.append("This shares some DNA, but it is more of a directional recommendation.")

    return " ".join(parts)


# =========================================================
# STYLES
# =========================================================
st.markdown(
    """
<style>
.stAppHeader { display:none; }
[data-testid="stHeader"] { display:none; }
[data-testid="stToolbar"] { visibility:hidden; height:0; position:fixed; }
[data-testid="stDecoration"] { display:none; }
[data-testid="stStatusWidget"] { display:none; }
#MainMenu { visibility:hidden; }
footer { visibility:hidden; }

.block-container{
  max-width: 860px;
  padding-top: 5.25rem;
  padding-bottom: 2rem;
}

.element-container:has(input[aria-label=""]) {
  display: none !important;
}

.search-card, .add-card, .result-card, .exact-card{
  border:1px solid rgba(148,163,184,0.28);
  border-radius:18px;
  background: rgba(255,255,255,0.03);
}
.search-card, .add-card{
  padding:14px;
}
.result-card, .exact-card{
  padding:16px;
  margin-bottom:12px;
}
.exact-card{
  border-color: rgba(34,197,94,0.35);
  background: rgba(34,197,94,0.08);
}

.score-scale-card{
  border:1px solid rgba(148,163,184,0.28);
  border-radius:16px;
  padding:12px;
  background: rgba(255,255,255,0.04);
  margin-bottom: 12px;
}
.score-scale-title{
  font-weight:700;
  margin-bottom:8px;
}
.score-scale-row{
  display:flex;
  gap:8px;
  flex-wrap:wrap;
}
.pill{
  padding:7px 12px;
  border-radius:999px;
  font-weight:800;
  font-size:0.92rem;
  display:inline-block;
}
.pill-green{background:#16a34a;color:#fff}
.pill-blue{background:#60a5fa;color:#fff}
.pill-yellow{background:#fde68a;color:#111827}

.score-card{
  border:1px solid rgba(148,163,184,0.28);
  border-radius:16px;
  padding:14px;
  background: rgba(255,255,255,0.04);
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  margin:8px 0 8px 0;
}
.score-main{
  font-weight:900;
  font-size:1.18rem;
  line-height:1.15;
}
.score-sub{
  font-size:0.82rem;
  color:#94a3b8;
  margin-top:4px;
}
.score-badge{
  padding:8px 14px;
  border-radius:999px;
  font-weight:800;
  font-size:0.95rem;
  white-space:nowrap;
}

.summary-line{
  margin-top:4px;
  color:#111827;
  font-size:0.96rem;
}
.summary-line b{
  color:inherit;
}

.why-line{
  color:#111827;
  line-height:1.5;
}

.missing-line{
  color:#9ca3af;
  font-style:italic;
  margin-top:6px;
}
.extra-line{
  color:#9ca3af;
  font-style:italic;
  margin-top:2px;
}
.small-help{
  color:#94a3b8;
  font-size:0.93rem;
  margin-top:-4px;
  margin-bottom:8px;
}

/* Light expanders with black text */
[data-testid="stExpander"] details{
  background:#ffffff !important;
  border:1px solid rgba(148,163,184,0.28) !important;
  border-radius:14px !important;
}

[data-testid="stExpander"] summary{
  background:#ffffff !important;
  color:#111827 !important;
  border-radius:14px !important;
  font-weight:600 !important;
}

[data-testid="stExpander"] summary:hover{
  background:#f8fafc !important;
  color:#111827 !important;
}

[data-testid="stExpanderDetails"]{
  background:#ffffff !important;
  color:#111827 !important;
  border-radius:0 0 14px 14px !important;
}

[data-testid="stExpanderDetails"] p,
[data-testid="stExpanderDetails"] div,
[data-testid="stExpanderDetails"] span,
[data-testid="stExpanderDetails"] label{
  color:#111827 !important;
}

@media (max-width: 768px){
  .block-container{
    padding-top: 4.75rem;
    padding-bottom: 1.25rem;
  }
  .score-card{
    flex-direction:column;
    align-items:flex-start;
    justify-content:flex-start;
    gap:10px;
  }
  .score-main{
    font-size:1.35rem;
  }
  .score-badge{
    align-self:flex-start;
    font-size:0.98rem;
    padding:9px 14px;
  }
}
</style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SEARCH VIEW
# =========================================================
if st.session_state.view == "search":
    st.title("Find your Chogan Perfume")
    st.subheader("Search")

    with st.container():
        st.markdown('<div class="search-card">', unsafe_allow_html=True)

        mode = st.radio("Choose input type:", ["By perfume name", "By notes only"])

        perfume_name = ""
        brand_name = ""

        if mode == "By perfume name":
            perfume_name = st.text_input("Perfume name (e.g., Nina)")
            brand_name = st.text_input("Brand (optional)")
        else:
            st.markdown(
                "<div class='small-help'>Try notes like <i>vanilla, patchouli, orange blossom</i> or <i>neroli, bergamot, musk</i>.</div>",
                unsafe_allow_html=True,
            )

        notes_text = st.text_input(
            "Desired notes (comma-separated)",
            placeholder="e.g., jasmine, vanilla, patchouli"
        )

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

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Search", use_container_width=True):
                st.session_state.last_query = {
                    "mode": mode,
                    "perfume_name": perfume_name,
                    "brand_name": brand_name,
                    "notes_text": notes_text,
                    "gender_choice": gender_choice,
                    "top_n": top_n,
                }
                st.session_state.view = "results"
                st.rerun()

        with c2:
            if st.button("Reset", use_container_width=True):
                reset_search()
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("###")
    if st.button("Add a new perfume to the database", use_container_width=True):
        st.session_state.view = "add"
        st.rerun()

# =========================================================
# RESULTS VIEW
# =========================================================
elif st.session_state.view == "results":
    qd = st.session_state.last_query

    if not qd:
        st.warning("No search found.")
        if st.button("Go to Search"):
            st.session_state.view = "search"
            st.rerun()
        st.stop()

    mode = qd.get("mode", "By perfume name")
    perfume_name = qd.get("perfume_name", "")
    brand_name = qd.get("brand_name", "")
    notes_text = qd.get("notes_text", "")
    gender_choice = qd.get("gender_choice", "Any")
    top_n = int(qd.get("top_n", 3))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back to Search", use_container_width=True):
            st.session_state.view = "search"
            st.rerun()
    with c2:
        if st.button("Add a new perfume to the database", use_container_width=True):
            st.session_state.view = "add"
            st.rerun()

    st.title("My Recommendations")
    score_scale_card()

    with st.spinner("Finding your best matches..."):
        raw_notes = normalize_notes_list(split_notes(notes_text))
        query_notes = set(raw_notes)

        used_pyramid = False
        query_top, query_heart, query_base = set(), set(), set()

        exact_hits = chogan.iloc[0:0].copy()
        exact_hit_rows = []

        if mode == "By perfume name" and perfume_name.strip():
            qname = perfume_name.strip()

            for idx, row in chogan.iterrows():
                insp = str(row.get("Inspiration", "")).strip()
                if not insp:
                    continue

                sim = name_similarity(qname, insp)

                if brand_name.strip():
                    b = norm_text(brand_name)
                    if b and b in norm_text(insp):
                        sim += 0.06

                contains = norm_text(qname) in norm_text(insp)

                if sim >= EXACT_MATCH_SIM_THRESHOLD or contains:
                    exact_hit_rows.append((sim, idx))

            exact_hit_rows.sort(key=lambda x: x[0], reverse=True)
            exact_idxs = [idx for _, idx in exact_hit_rows[:10]]
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

                st.markdown('<div class="exact-card">', unsafe_allow_html=True)
                st.markdown(f"### ✅ Exact match — **{ref}**")
                st.write(f"Inspiration: *{hit.get('Inspiration','')}*")

                with st.expander("View notes pyramid", expanded=False):
                    st.write(f"Top: {hit.get('Top Notes','')}")
                    st.write(f"Heart: {hit.get('Heart Notes','')}")
                    st.write(f"Base: {hit.get('Base Notes','')}")

                st.markdown("</div>", unsafe_allow_html=True)

        used_external = None
        if mode == "By perfume name" and perfume_name.strip() and (external is not None) and (not external.empty):
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

                score, matched_notes, matched_pillars, missing_notes, extra_notes = score_perfume(
                    query_notes=query_notes,
                    row=row,
                    used_pyramid=used_pyramid,
                    query_top=query_top,
                    query_heart=query_heart,
                    query_base=query_base,
                )

                if mode == "By perfume name" and perfume_name.strip():
                    sim = name_similarity(perfume_name, str(row.get("Inspiration", "")))
                    if sim > 0.88:
                        score = min(score + 0.6, 10.0)
                    elif sim > 0.75:
                        score = min(score + 0.3, 10.0)

                results.append((score, matched_notes, matched_pillars, missing_notes, extra_notes, row))

            results.sort(key=lambda x: x[0], reverse=True)

            shown = 0
            for score, matched_notes, matched_pillars, missing_notes, extra_notes, row in results:
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

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f"### #{shown} — **{ref}**")
                st.write(f"Inspiration: *{row.get('Inspiration','')}*")

                label, bg, fg = score_badge(score)
                st.markdown(
                    f"""
<div class="score-card">
  <div>
    <div class="score-main">Match score: {score:.2f} / 10</div>
    <div class="score-sub">Higher scores mean closer note + accord alignment</div>
  </div>
  <div class="score-badge" style="background:{bg}; color:{fg};">{label}</div>
</div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    compact_summary(matched_notes, matched_pillars, missing_notes),
                    unsafe_allow_html=True,
                )

                notes_txt = ", ".join(matched_notes) if matched_notes else "None"
                accords_txt = ", ".join(matched_pillars) if matched_pillars else "None"
                why_txt = why_this_match_text(score, matched_notes, matched_pillars, missing_notes)

                with st.expander("Why this match?"):
                    st.markdown(f"<div class='why-line'><b>Matched notes:</b> {notes_txt}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='why-line'><b>Matched accords:</b> {accords_txt}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='why-line'>{why_txt}</div>", unsafe_allow_html=True)

                with st.expander("View notes pyramid"):
                    st.write(f"Top: {row.get('Top Notes','')}")
                    st.write(f"Heart: {row.get('Heart Notes','')}")
                    st.write(f"Base: {row.get('Base Notes','')}")

                    if missing_notes:
                        miss_txt = ", ".join(missing_notes)
                        st.markdown(f"<div class='missing-line'>Missing notes: {miss_txt}</div>", unsafe_allow_html=True)

                    if extra_notes:
                        extra_txt = ", ".join(extra_notes)
                        suffix = "…" if len(extra_notes) == 10 else ""
                        st.markdown(f"<div class='extra-line'>Other notes present: {extra_txt}{suffix}</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                if shown >= top_n:
                    break

            if shown == 0:
                st.warning(
                    "Sorry, we don't have a good match for the notes in the perfume you are looking for. "
                    "Would you like to try something else?"
                )

# =========================================================
# ADD VIEW
# =========================================================
elif st.session_state.view == "add":
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back to Search", use_container_width=True):
            st.session_state.view = "search"
            st.rerun()
    with c2:
        if st.button("Go to Results", use_container_width=True):
            st.session_state.view = "results"
            st.rerun()

    st.title("Add a new perfume to the database")

    with st.container():
        st.markdown('<div class="add-card">', unsafe_allow_html=True)

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

        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        if not new_perfume.strip():
            st.error("Perfume name required.")
        else:
            try:
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
                upsert_external_to_sheets(row_dict)
                st.success("Saved.")
            except Exception as e:
                st.error(f"Could not save to Google Sheets: {e}")

    try:
        external_latest = load_external_from_sheets_cached()
    except Exception:
        external_latest = external

    with st.expander("View saved external perfumes"):
        st.dataframe(external_latest.tail(50), use_container_width=True)
