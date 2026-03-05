import streamlit as st
import pandas as pd
import re
import json
import gspread
import unicodedata
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher

# =========================================================
# PAGE CONFIG
# =========================================================
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

NOTE_WEIGHT = 0.70
VIBE_WEIGHT = 0.30
DNA_BOOST = 0.7
MIN_SCORE_TO_SHOW = 3.0


# =========================================================
# VIEW STATE (Search vs Results)
# =========================================================

if "view" not in st.session_state:
    st.session_state.view = "search"  # "search" or "results"
if "last_payload_key" not in st.session_state:
    st.session_state.last_payload_key = None
if "cached_results" not in st.session_state:
    st.session_state.cached_results = None
if "scroll_nonce" not in st.session_state:
    st.session_state.scroll_nonce = 0

_defaults = {
    "mode": "By perfume name",
    "perfume_name": "",
    "brand_name": "",
    "notes_text": "",
    "family_filter": "",
    "gender_choice": "Any",
    "top_n": 3,
    "mobile_mode": True,  # default ON; user can turn off
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_search():
    for k, v in _defaults.items():
        st.session_state[k] = v
    st.session_state.view = "search"
    st.session_state.last_payload_key = None
    st.session_state.cached_results = None
    st.session_state.scroll_nonce += 1


def go_to_search():
    st.session_state.view = "search"
    st.session_state.scroll_nonce += 1


def go_to_results():
    st.session_state.view = "results"
    st.session_state.scroll_nonce += 1


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

    key_perfume = row_dict.get("Perfume", "").strip().lower()
    key_brand = row_dict.get("Brand", "").strip().lower()

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

def strip_accents(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c))


def norm_text(s):
    s = strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def name_similarity(a, b):
    a = norm_text(a)
    b = norm_text(b)

    fuzz = SequenceMatcher(None, a, b).ratio()
    ta = set(a.split())
    tb = set(b.split())
    jac = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0
    return (fuzz + jac) / 2.0


# =========================================================
# NOTES HELPERS
# =========================================================

def split_notes(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    parts = re.split(r"[,/;]+", str(x))
    return [p.strip().lower() for p in parts if p.strip()]


def normalize_note(n):
    return strip_accents(str(n).strip().lower())


def normalize_notes_list(lst):
    return [normalize_note(x) for x in lst if str(x).strip()]


# =========================================================
# PILLARS (VIBE)
# =========================================================

PILLARS = {
    "fruity": {
        "pear", "raspberry", "strawberry", "lychee", "blackcurrant", "currant", "peach", "plum",
        "apple", "mango", "orange", "mandarin", "tangerine", "bergamot", "lemon"
    },
    "floral": {"rose", "black rose", "jasmine", "tuberose", "orange blossom", "peony", "datura", "iris", "violet", "orchid"},
    "gourmand": {"vanilla", "praline", "caramel", "coffee", "tonka", "chocolate", "benzoin", "toffee", "cocoa"},
    "woody": {"patchouli", "cedar", "cedarwood", "sandalwood", "vetiver", "moss", "oakmoss", "papyrus"},
    "musky": {"musk", "white musk", "ambroxan", "ambergris"},
    "resinous": {"incense", "labdanum", "amber", "myrrh"},
}

ANCHOR_COMBOS = [
    ({"rose", "patchouli"}, 0.8),
    ({"vanilla", "patchouli"}, 0.6),
    ({"coffee", "vanilla"}, 0.8),
    ({"praline", "vanilla"}, 0.7),
]


def detect_pillars(notes_set):
    blob = " ".join(sorted(notes_set))
    found = set()
    for pillar, kws in PILLARS.items():
        for kw in kws:
            if kw in blob:
                found.add(pillar)
                break
    return found


def penalties(query_pillars, perfume_pillars):
    pen = 0.0
    if "gourmand" in query_pillars and "gourmand" not in perfume_pillars:
        pen -= 1.0
    return pen


# =========================================================
# SCORING (PYRAMID + VIBE) + MATCH EXPLANATIONS
# =========================================================

def get_row_note_sets(row):
    top = set(normalize_notes_list(split_notes(row.get("Top Notes", ""))))
    heart = set(normalize_notes_list(split_notes(row.get("Heart Notes", ""))))
    base = set(normalize_notes_list(split_notes(row.get("Base Notes", ""))))
    all_notes = top | heart | base
    return top, heart, base, all_notes


def score_notes_pyramid(query_top, query_heart, query_base, perf_top, perf_heart, perf_base):
    q_top = set(query_top)
    q_heart = set(query_heart)
    q_base = set(query_base)

    denom = (2.0 * len(q_top)) + (1.6 * len(q_heart)) + (1.4 * len(q_base))
    if denom <= 0:
        return 0.0

    s = 0.0

    for n in q_top:
        if n in perf_top:
            s += 2.0
        elif n in perf_heart:
            s += 1.2
        elif n in perf_base:
            s += 0.8

    for n in q_heart:
        if n in perf_heart:
            s += 1.6
        elif n in perf_top:
            s += 1.2
        elif n in perf_base:
            s += 1.0

    for n in q_base:
        if n in perf_base:
            s += 1.4
        elif n in perf_heart:
            s += 1.1

    return max(min(s / denom, 1.0), 0.0)


def score_notes_simple(query_notes, perfume_notes):
    if not query_notes:
        return 0.0
    overlap = len(set(query_notes) & set(perfume_notes))
    return overlap / max(len(set(query_notes)), 1)


def score_perfume(query_notes, row, used_pyramid=False, query_top=None, query_heart=None, query_base=None):
    perf_top, perf_heart, perf_base, perf_all = get_row_note_sets(row)

    if used_pyramid and (query_top or query_heart or query_base):
        note_score = score_notes_pyramid(query_top, query_heart, query_base, perf_top, perf_heart, perf_base)
        q_all_for_vibe = set(query_top) | set(query_heart) | set(query_base)
    else:
        note_score = score_notes_simple(query_notes, perf_all)
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

    matched_notes = sorted(set(q_all_for_vibe) & set(perf_all))
    matched_pillars = sorted(query_pillars & perfume_pillars)

    return score10, matched_notes, matched_pillars


# =========================================================
# LOAD DATA
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
# FILTERS (ROBUST GENDER)
# =========================================================

def norm_gender(val: str) -> str:
    v = str(val).strip().upper()
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


def apply_filters(df: pd.DataFrame, family_filter: str, gender_choice: str) -> pd.DataFrame:
    filtered = df.copy()

    if family_filter.strip() and "Olfactory Family" in filtered.columns:
        filtered = filtered[
            filtered["Olfactory Family"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(family_filter.strip().lower(), na=False)
        ]

    if "Gender" in filtered.columns:
        g = filtered["Gender"].fillna("").astype(str).apply(norm_gender)

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

    return filtered


# =========================================================
# SEARCH / RECOMMENDATION PIPELINE
# =========================================================

def get_ref(row) -> str:
    ref = (
        row.get("Perfume reference")
        or row.get("Perfume ref.")
        or row.get("Reference")
        or row.get("Code")
        or row.get("ID")
        or ""
    )
    return str(ref).strip()


def score_key_block():
    st.markdown(
        """
**Score key**

- **7.0–10.0** → Excellent match  
- **5.0–6.9** → Good match  
- **3.0–4.9** → Some overlap, but different  
        """
    )


def current_payload() -> dict:
    return {
        "mode": st.session_state.mode,
        "perfume_name": st.session_state.perfume_name,
        "brand_name": st.session_state.brand_name,
        "notes_text": st.session_state.notes_text,
        "family_filter": st.session_state.family_filter,
        "gender_choice": st.session_state.gender_choice,
        "top_n": int(st.session_state.top_n),
    }


def payload_key(p: dict) -> str:
    return json.dumps(p, sort_keys=True, ensure_ascii=False)


def compute_results(payload: dict) -> dict:
    mode = payload["mode"]
    perfume_name = payload["perfume_name"]
    brand_name = payload["brand_name"]
    notes_text = payload["notes_text"]
    family_filter = payload["family_filter"]
    gender_choice = payload["gender_choice"]
    top_n = payload["top_n"]

    raw_notes = normalize_notes_list(split_notes(notes_text))
    query_notes = set(raw_notes)

    used_pyramid = False
    query_top, query_heart, query_base = set(), set(), set()

    direct_hits = chogan.iloc[0:0]
    if mode == "By perfume name" and perfume_name.strip():
        q = perfume_name.strip().lower()
        direct_hits = chogan[
            chogan["Inspiration"].fillna("").astype(str).str.lower().str.contains(q, na=False)
        ]
        if brand_name.strip() and len(direct_hits) > 1:
            bq = brand_name.strip().lower()
            direct_hits = direct_hits[
                direct_hits["Inspiration"].fillna("").astype(str).str.lower().str.contains(bq, na=False)
            ]

    used_external = None
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

    if (not query_notes) and len(direct_hits) > 0:
        seed = direct_hits.iloc[0].to_dict()
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

    filtered = apply_filters(chogan, family_filter, gender_choice)

    direct_refs = set()
    if len(direct_hits) > 0:
        for _, hit in direct_hits.iterrows():
            direct_refs.add(get_ref(hit).lower())

    recommendations = []
    if query_notes or used_pyramid:
        for _, row in filtered.iterrows():
            ref = get_ref(row)
            if ref.strip().lower() in direct_refs:
                continue

            score, matched_notes, matched_pillars = score_perfume(
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
                    score = min(score + 1.5, 10.0)
                elif sim > 0.70:
                    score = min(score + 0.8, 10.0)

            recommendations.append({
                "score": float(score),
                "matched_notes": matched_notes,
                "matched_pillars": matched_pillars,
                "row": row,
            })

        recommendations.sort(key=lambda d: d["score"], reverse=True)
        recommendations = [r for r in recommendations if r["score"] >= MIN_SCORE_TO_SHOW][:top_n]

    return {
        "direct_hits": direct_hits,
        "used_external": used_external,
        "recommendations": recommendations,
    }


# =========================================================
# SCROLL-TO-TOP
# =========================================================

def scroll_to_top():
    st.markdown(
        f"""
<script>
(function() {{
  const nonce = {st.session_state.scroll_nonce};
  if (!window.__scrollNonce || window.__scrollNonce !== nonce) {{
    window.__scrollNonce = nonce;
    window.scrollTo(0, 0);
  }}
}})();
</script>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# UI
# =========================================================

st.title("Find your Chogan Perfume")

tab_search, tab_add = st.tabs(["Search", "Add a new perfume to the database"])

# ---------------------------------------------------------
# SEARCH TAB
# ---------------------------------------------------------
with tab_search:
    st.toggle("Mobile mode (results replace search)", key="mobile_mode")

    if st.session_state.view == "results":
        scroll_to_top()

    # ===== SEARCH VIEW =====
    if st.session_state.view == "search":
        if st.session_state.mobile_mode:
            st.subheader("Search")

            # Keep the always-visible inputs minimal (less keyboard takeover)
            st.radio("Choose input type:", ["By perfume name", "By notes only"], key="mode")

            if st.session_state.mode == "By perfume name":
                st.text_input("Perfume name (e.g., Nina)", key="perfume_name")
                st.text_input("Brand (optional)", key="brand_name")
            else:
                st.text_input("Desired notes (comma-separated)", key="notes_text")

            # Put long inputs + filters behind expanders (better mobile)
            with st.expander("Notes (optional)", expanded=False):
                st.text_input("Desired notes (comma-separated)", key="notes_text")

            with st.expander("Filters (optional)", expanded=False):
                st.text_input("Olfactory family contains", key="family_filter")
                st.selectbox(
                    "Gender preference",
                    [
                        "Any",
                        "Women (F)",
                        "Men (M)",
                        "Unisex (U)",
                        "Women or Unisex (F/U)",
                        "Men or Unisex (M/U)",
                    ],
                    key="gender_choice",
                )
                st.slider("How many recommendations?", 1, 5, key="top_n")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Search", type="primary", use_container_width=True):
                    p = current_payload()
                    k = payload_key(p)
                    st.session_state.last_payload_key = k
                    st.session_state.cached_results = compute_results(p)
                    go_to_results()
            with c2:
                st.button("Reset", on_click=reset_search, use_container_width=True)

        else:
            # Desktop: form left, placeholder right until searched
            left, right = st.columns([1, 2])

            with left:
                st.subheader("Search")

                st.radio("Choose input type:", ["By perfume name", "By notes only"], key="mode")

                if st.session_state.mode == "By perfume name":
                    st.text_input("Perfume name (e.g., Nina)", key="perfume_name")
                    st.text_input("Brand (optional)", key="brand_name")
                else:
                    st.text_input("Perfume name (optional)", key="perfume_name")
                    st.text_input("Brand (optional)", key="brand_name")

                st.text_input("Desired notes (comma-separated)", key="notes_text")

                st.subheader("Filters (optional)")
                st.text_input("Olfactory family contains", key="family_filter")

                st.selectbox(
                    "Gender preference",
                    [
                        "Any",
                        "Women (F)",
                        "Men (M)",
                        "Unisex (U)",
                        "Women or Unisex (F/U)",
                        "Men or Unisex (M/U)",
                    ],
                    key="gender_choice",
                )

                st.slider("How many recommendations?", 1, 5, key="top_n")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Search", type="primary", use_container_width=True):
                        p = current_payload()
                        k = payload_key(p)
                        st.session_state.last_payload_key = k
                        st.session_state.cached_results = compute_results(p)
                        go_to_results()
                with c2:
                    st.button("Reset", on_click=reset_search, use_container_width=True)

            with right:
                st.info("Run a search to see recommendations.")

    # ===== RESULTS VIEW =====
    else:
        p = current_payload()
        k = payload_key(p)
        if st.session_state.cached_results is None or st.session_state.last_payload_key != k:
            st.session_state.last_payload_key = k
            st.session_state.cached_results = compute_results(p)

        results = st.session_state.cached_results
        direct_hits = results["direct_hits"]
        recs = results["recommendations"]
        used_external = results["used_external"]

        if st.session_state.mobile_mode:
            # Mobile results replace search
            topbar_left, topbar_right = st.columns([1, 1])
            with topbar_left:
                st.subheader("My Recommendations")
            with topbar_right:
                if st.button("← Back", use_container_width=True):
                    go_to_search()

            if used_external is not None:
                st.caption(f"Using saved notes for: {used_external.get('Perfume','')} ({used_external.get('Brand','')})")

            if len(direct_hits) > 0:
                st.success(f"Direct match found in Chogan inspirations ({len(direct_hits)} result(s)).")
                for rank, (_, hit) in enumerate(direct_hits.head(int(st.session_state.top_n)).iterrows(), start=1):
                    ref = get_ref(hit)
                    st.markdown(f"### ✅ Direct match #{rank} — **{ref}**")
                    st.write(f"Inspiration: *{hit.get('Inspiration','')}*")
                    st.write(f"Top: {hit.get('Top Notes','')}")
                    st.write(f"Heart: {hit.get('Heart Notes','')}")
                    st.write(f"Base: {hit.get('Base Notes','')}")
                    st.divider()

            if not recs:
                st.warning(
                    "Sorry, we don't have a good match for the notes in the perfume you are looking for. "
                    "Would you like to try something else?"
                )
            else:
                for i, item in enumerate(recs, start=1):
                    row = item["row"]
                    score = item["score"]
                    matched_notes = item["matched_notes"]
                    matched_pillars = item["matched_pillars"]

                    ref = get_ref(row)

                    st.markdown(f"### #{i} — **{ref}**")
                    st.write(f"Inspiration: *{row.get('Inspiration','')}*")
                    st.write(f"Top: {row.get('Top Notes','')}")
                    st.write(f"Heart: {row.get('Heart Notes','')}")
                    st.write(f"Base: {row.get('Base Notes','')}")
                    st.write(f"**Matched notes:** {', '.join(matched_notes) if matched_notes else 'None'}")
                    st.write(f"**Matched accords:** {', '.join(matched_pillars) if matched_pillars else 'None'}")
                    st.write(f"**Match score:** {score:.2f} / 10")
                    st.divider()

            # ✅ Score key at the bottom (requested)
            st.divider()
            score_key_block()

        else:
            # Desktop: show search left, results right
            left, right = st.columns([1, 2])

            with left:
                st.subheader("Search")

                st.radio("Choose input type:", ["By perfume name", "By notes only"], key="mode")

                if st.session_state.mode == "By perfume name":
                    st.text_input("Perfume name (e.g., Nina)", key="perfume_name")
                    st.text_input("Brand (optional)", key="brand_name")
                else:
                    st.text_input("Perfume name (optional)", key="perfume_name")
                    st.text_input("Brand (optional)", key="brand_name")

                st.text_input("Desired notes (comma-separated)", key="notes_text")

                st.subheader("Filters (optional)")
                st.text_input("Olfactory family contains", key="family_filter")

                st.selectbox(
                    "Gender preference",
                    [
                        "Any",
                        "Women (F)",
                        "Men (M)",
                        "Unisex (U)",
                        "Women or Unisex (F/U)",
                        "Men or Unisex (M/U)",
                    ],
                    key="gender_choice",
                )

                st.slider("How many recommendations?", 1, 5, key="top_n")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Search", type="primary", use_container_width=True, key="search_again"):
                        p2 = current_payload()
                        k2 = payload_key(p2)
                        st.session_state.last_payload_key = k2
                        st.session_state.cached_results = compute_results(p2)
                        go_to_results()
                with c2:
                    st.button("Reset", on_click=reset_search, use_container_width=True, key="reset2")

                st.divider()
                score_key_block()

            with right:
                st.subheader("My Recommendations")

                if used_external is not None:
                    st.caption(f"Using saved notes for: {used_external.get('Perfume','')} ({used_external.get('Brand','')})")

                if len(direct_hits) > 0:
                    st.success(f"Direct match found in Chogan inspirations ({len(direct_hits)} result(s)).")
                    for rank, (_, hit) in enumerate(direct_hits.head(int(st.session_state.top_n)).iterrows(), start=1):
                        ref = get_ref(hit)
                        st.markdown(f"### ✅ Direct match #{rank} — **{ref}**")
                        st.write(f"Inspiration: *{hit.get('Inspiration','')}*")
                        st.write(f"Top: {hit.get('Top Notes','')}")
                        st.write(f"Heart: {hit.get('Heart Notes','')}")
                        st.write(f"Base: {hit.get('Base Notes','')}")
                        st.divider()

                if not recs:
                    st.warning(
                        "Sorry, we don't have a good match for the notes in the perfume you are looking for. "
                        "Would you like to try something else?"
                    )
                else:
                    for i, item in enumerate(recs, start=1):
                        row = item["row"]
                        score = item["score"]
                        matched_notes = item["matched_notes"]
                        matched_pillars = item["matched_pillars"]

                        ref = get_ref(row)

                        st.markdown(f"### #{i} — **{ref}**")
                        st.write(f"Inspiration: *{row.get('Inspiration','')}*")
                        st.write(f"**Match score:** {score:.2f} / 10")
                        st.write(f"**Matched notes:** {', '.join(matched_notes) if matched_notes else 'None'}")
                        st.write(f"**Matched accords:** {', '.join(matched_pillars) if matched_pillars else 'None'}")
                        st.write(f"Top: {row.get('Top Notes','')}")
                        st.write(f"Heart: {row.get('Heart Notes','')}")
                        st.write(f"Base: {row.get('Base Notes','')}")
                        st.divider()


# ---------------------------------------------------------
# ADD NEW PERFUME TAB
# ---------------------------------------------------------
with tab_add:
    st.subheader("Add / Update an External Perfume")

    if external_ws is None:
        st.error("Google Sheets is not connected. Fix your credentials/secrets first.")
    else:
        with st.form("add_external", clear_on_submit=True):
            c1, c2 = st.columns(2)

            with c1:
                new_perfume = st.text_input("Perfume")
                new_brand = st.text_input("Brand")
                new_family = st.text_input("Olfactory Family")
                new_gender = st.selectbox("Gender", ["", "F", "M", "U"])

            with c2:
                new_top = st.text_input("Top Notes (comma-separated)")
                new_heart = st.text_input("Heart Notes (comma-separated)")
                new_base = st.text_input("Base Notes (comma-separated)")
                new_all = st.text_input("All Notes (comma-separated) — use if no pyramid")

            submitted = st.form_submit_button("Save perfume")

        if submitted:
            if not new_perfume.strip():
                st.error("Perfume name required.")
            else:
                row_dict = {
                    "Perfume": new_perfume.strip(),
                    "Brand": new_brand.strip(),
                    "Gender": new_gender.strip(),
                    "Top Notes": new_top.strip(),
                    "Heart Notes": new_heart.strip(),
                    "Base Notes": new_base.strip(),
                    "All Notes": new_all.strip(),
                    "Olfactory Family": new_family.strip(),
                }

                upsert_external_to_sheets(external_ws, row_dict)
                st.success("Saved (updated if already existed).")

                try:
                    external, external_ws = load_external_from_sheets()
                except Exception:
                    pass

    with st.expander("View saved external perfumes"):
        st.dataframe(external.tail(50))
