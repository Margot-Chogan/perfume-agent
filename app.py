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
        "pear","raspberry","strawberry","lychee","blackcurrant","currant","peach","plum",
        "apple","mango","orange","mandarin","tangerine","bergamot","lemon"
    },
    "floral": {"rose","black rose","jasmine","tuberose","orange blossom","peony","datura","iris","violet","orchid"},
    "gourmand": {"vanilla","praline","caramel","coffee","tonka","chocolate","benzoin","toffee","cocoa"},
    "woody": {"patchouli","cedar","cedarwood","sandalwood","vetiver","moss","oakmoss","papyrus"},
    "musky": {"musk","white musk","ambroxan","ambergris"},
    "resinous": {"incense","labdanum","amber","myrrh"},
}

ANCHOR_COMBOS = [
    ({"rose","patchouli"}, 0.8),
    ({"vanilla","patchouli"}, 0.6),
    ({"coffee","vanilla"}, 0.8),
    ({"praline","vanilla"}, 0.7),
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
# SCORING (PYRAMID + VIBE)
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
# UI HELPERS
# =========================================================
def score_label(score_10: float) -> str:
    if score_10 >= 7.0:
        return "Excellent match"
    if score_10 >= 5.0:
        return "Good match"
    if score_10 >= 3.0:
        return "Worth a try"
    return "Low match"


def score_badge_bg(score_10: float) -> str:
    if score_10 >= 7.0:
        return "#16a34a"  # green
    if score_10 >= 5.0:
        return "#60a5fa"  # light blue
    if score_10 >= 3.0:
        return "#fde68a"  # light yellow
    return "#9ca3af"      # gray


def score_badge_text(score_10: float) -> str:
    if 3.0 <= score_10 < 5.0:
        return "#111827"
    return "#ffffff"


def score_key_card():
    st.markdown(
        """
        <div style="border:1px solid #e5e7eb; border-radius:16px; padding:14px; margin:0 0 12px 0; background:#ffffff;">
          <div style="font-weight:800; font-size:1.05rem; margin-bottom:10px;">How to read the match score</div>
          <div style="display:flex; flex-wrap:wrap; gap:10px;">
            <span style="padding:8px 12px; border-radius:999px; background:#16a34a; color:#fff; font-weight:800;">
              7.0–10.0 — Excellent match
            </span>
            <span style="padding:8px 12px; border-radius:999px; background:#60a5fa; color:#fff; font-weight:800;">
              5.0–6.9 — Good match
            </span>
            <span style="padding:8px 12px; border-radius:999px; background:#fde68a; color:#111827; font-weight:800;">
              3.0–4.9 — Worth a try
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_card(score_10: float):
    bg = score_badge_bg(score_10)
    tc = score_badge_text(score_10)
    label = score_label(score_10)

    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb; border-radius:16px; padding:14px; margin:10px 0 8px 0; background:#ffffff;">
          <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
            <div style="font-weight:900; font-size:1.45rem;">
              Match score: <span style="font-size:2.0rem; font-weight:1000;">{score_10:.2f}</span> / 10
            </div>
            <div style="padding:8px 12px; border-radius:999px; font-weight:900; color:{tc}; background:{bg}; white-space:nowrap;">
              {label}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def matched_line(matched_notes: list[str], matched_pillars: list[str]):
    notes_txt = ", ".join(matched_notes) if matched_notes else "None"
    pillars_txt = ", ".join(matched_pillars) if matched_pillars else "None"
    st.write(f"**Matched notes:** {notes_txt}   |   **Matched accords:** {pillars_txt}")

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
# APP
# =========================================================
tab_search, tab_add = st.tabs(["Search", "Add a new perfume to the database"])

with tab_search:
    st.title("Find your Chogan Perfume")

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Search Mode")

        with st.form("search_form"):
            mode = st.radio("Choose input type:", ["By perfume name", "By notes only"])

            perfume_name = ""
            brand_name = ""

            if mode == "By perfume name":
                perfume_name = st.text_input("Perfume name (e.g., Nina)")
                brand_name = st.text_input("Brand (optional)")

            notes_text = st.text_input("Desired notes (comma-separated)")

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
            search_clicked = st.form_submit_button("Search")

    with right:
        st.subheader("My Recommendations")
        score_key_card()  # always visible at top

        if not search_clicked:
            st.info("Click **Search** to run recommendations.")
        else:
            # 1) typed notes
            raw_notes = normalize_notes_list(split_notes(notes_text))
            query_notes = set(raw_notes)

            used_pyramid = False
            query_top, query_heart, query_base = set(), set(), set()

            # 2) exact match (inspiration)
            direct_hits = chogan.iloc[0:0]
            if mode == "By perfume name" and perfume_name.strip():
                q = perfume_name.strip().lower()
                direct_hits = chogan[chogan["Inspiration"].fillna("").astype(str).str.lower().str.contains(q, na=False)]
                if brand_name.strip() and len(direct_hits) > 1:
                    bq = brand_name.strip().lower()
                    direct_hits = direct_hits[
                        direct_hits["Inspiration"].fillna("").astype(str).str.lower().str.contains(bq, na=False)
                    ]

            if len(direct_hits) > 0:
                st.success(f"Exact match found in Chogan inspirations ({len(direct_hits)} result(s)).")
                hit = direct_hits.iloc[0]
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

            # 3) external DB notes
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

            # 4) seed from exact match if still no notes
            if (not query_notes) and len(direct_hits) > 0:
                seed = direct_hits.iloc[0].to_dict()
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

            # 5) gender filter
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

            # 6) recommendations
            if not query_notes and not used_pyramid:
                st.warning("Add some notes, or search a perfume name that exists in your external database.")
            else:
                direct_refs = set()
                if len(direct_hits) > 0:
                    for _, hit in direct_hits.iterrows():
                        ref = (
                            hit.get("Perfume reference")
                            or hit.get("Perfume ref.")
                            or hit.get("Reference")
                            or hit.get("Code")
                            or hit.get("ID")
                            or ""
                        )
                        direct_refs.add(str(ref).strip().lower())

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
                    if str(ref).strip().lower() in direct_refs:
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

                    results.append((score, matched_notes, matched_pillars, row))

                results.sort(key=lambda x: x[0], reverse=True)

                shown = 0
                for score, matched_notes, matched_pillars, row in results:
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
                    score_card(score)
                    matched_line(matched_notes, matched_pillars)

                    st.write(f"Top: {row.get('Top Notes','')}")
                    st.write(f"Heart: {row.get('Heart Notes','')}")
                    st.write(f"Base: {row.get('Base Notes','')}")
                    st.divider()

                    if shown >= top_n:
                        break

                if shown == 0:
                    st.warning(
                        "Sorry, we don't have a good match for the notes in the perfume you are looking for. "
                        "Would you like to try something else?"
                    )

with tab_add:
    st.subheader("Add / Update an External Perfume (manual entry)")

    with st.form("add_external", clear_on_submit=True):
        c1, c2 = st.columns(2)

        with c1:
            new_perfume = st.text_input("Perfume")
            new_brand = st.text_input("Brand")
            new_family = st.text_input("Olfactory Family")
            new_gender = st.selectbox("Gender", ["", "F", "M", "U"])

        with c2:
            new_top = st.text_input("Top Notes")
            new_heart = st.text_input("Heart Notes")
            new_base = st.text_input("Base Notes")
            new_all = st.text_input("All Notes")

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
                "Olfactory Family": new_family.strip(),
            }
            upsert_external_to_sheets(external_ws, row_dict)
            st.success("Saved.")
            st.rerun()

    with st.expander("View saved external perfumes"):
        st.dataframe(external.tail(50))
