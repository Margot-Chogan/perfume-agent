import streamlit as st
import pandas as pd
import re
import json
import gspread
import unicodedata
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher

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

NOTE_WEIGHT = 0.65
VIBE_WEIGHT = 0.35
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

    jac = len(ta & tb) / len(ta | tb) if (ta or tb) else 0

    return (fuzz + jac) / 2


# =========================================================
# NOTES HELPERS
# =========================================================

def split_notes(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    parts = re.split(r"[,/;]+", str(x))
    return [p.strip().lower() for p in parts if p.strip()]


def normalize_note(n):
    n = strip_accents(str(n).strip().lower())
    return n


# =========================================================
# PILLARS
# =========================================================

PILLARS = {
    "fruity": {"pear","raspberry","strawberry","lychee","blackcurrant","peach","plum","apple","mango","orange","mandarin","tangerine","bergamot","lemon"},
    "floral": {"rose","black rose","jasmine","tuberose","orange blossom","peony","datura","iris","violet","orchid","vanilla orchid","neroli"},
    "gourmand": {"vanilla","praline","caramel","coffee","tonka","tonka bean","chocolate","benzoin","toffee","cocoa","honey","sugar"},
    "woody": {"patchouli","cedar","cedarwood","sandalwood","vetiver","moss","oakmoss","papyrus","woods","woody"},
    "musky": {"musk","white musk","ambroxan","ambergris","ambrox"},
    "resinous": {"incense","labdanum","amber","resin"},
}


def detect_pillars(notes):
    blob = " ".join(notes)
    found = set()
    for pillar, kws in PILLARS.items():
        for kw in kws:
            if kw in blob:
                found.add(pillar)
                break
    return found


# =========================================================
# NOTE RARITY WEIGHTS
# (rare notes increase similarity impact)
# =========================================================

NOTE_RARITY = {
    "coffee": 2.0,
    "licorice": 2.2,
    "incense": 1.9,
    "praline": 1.6,
    "caramel": 1.5,
    "tonka": 1.4,
    "tonka bean": 1.4,
    "benzoin": 1.4,
    "patchouli": 1.3,
    "tuberose": 1.3,
    "datura": 1.3,
    "ambroxan": 1.2,

    # common notes (low weight)
    "bergamot": 0.6,
    "lemon": 0.6,
    "orange": 0.6,
    "musk": 0.7,
    "vanilla": 0.8,
    "rose": 0.9,
    "jasmine": 0.9,
}


# =========================================================
# ANCHOR COMBOS
# =========================================================

ANCHOR_COMBOS = [
    ({"rose","patchouli"}, 0.8),
    ({"vanilla","patchouli"}, 0.6),
    ({"coffee","vanilla"}, 0.8),
    ({"praline","vanilla"}, 0.7),
]


# =========================================================
# PENALTIES
# =========================================================

def penalties(query_pillars, perfume_pillars):
    pen = 0
    # Example penalty: if user wants gourmand but perfume isn't gourmand
    if "gourmand" in query_pillars and "gourmand" not in perfume_pillars:
        pen -= 1
    return pen


# =========================================================
# SCORING
# =========================================================

def score_perfume(query_notes, row):
    top = set(normalize_note(n) for n in split_notes(row.get("Top Notes","")))
    heart = set(normalize_note(n) for n in split_notes(row.get("Heart Notes","")))
    base = set(normalize_note(n) for n in split_notes(row.get("Base Notes","")))
    perfume_notes = top | heart | base

    # Weighted overlap
    weighted_overlap = 0.0
    total_weight = 0.0

    for note in query_notes:
        w = NOTE_RARITY.get(note, 1.0)
        total_weight += w
        if note in perfume_notes:
            weighted_overlap += w

    note_score = weighted_overlap / max(total_weight, 1.0)  # 0..1

    query_pillars = detect_pillars(query_notes)
    perfume_pillars = detect_pillars(perfume_notes)

    pillar_overlap = len(query_pillars & perfume_pillars)
    vibe_score = pillar_overlap / max(len(query_pillars), 1)  # 0..1

    blended = NOTE_WEIGHT * note_score + VIBE_WEIGHT * vibe_score  # 0..1

    # Anchor bonuses
    for combo, b in ANCHOR_COMBOS:
        if combo <= query_notes and combo <= perfume_notes:
            blended += b / 10.0

    blended += penalties(query_pillars, perfume_pillars) / 10.0

    # DNA similarity boost
    if pillar_overlap >= 3:
        blended += DNA_BOOST / 10.0

    return max(min(blended * 10.0, 10.0), 0.0)


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
# UI
# =========================================================

st.title("Find your Chogan Perfume")

left, right = st.columns([1, 2])

with left:
    st.subheader("Search Mode")

    # ✅ Search runs ONLY when button is clicked
    with st.form("search_form"):
        mode = st.radio("Choose input type:", ["By perfume name", "By notes only"])

        perfume_name = ""
        brand_name = ""

        if mode == "By perfume name":
            perfume_name = st.text_input("Perfume name")
            brand_name = st.text_input("Brand (optional)")

        notes_text = st.text_input("Desired notes (comma-separated)")

        st.subheader("Filters (optional)")
        family_filter = st.text_input("Olfactory family contains (optional)")
        gender_choice = st.selectbox(
            "Gender preference",
            ["Any", "Women (F)", "Men (M)", "Unisex (U)"]
        )
        top_n = st.slider("How many recommendations?", 1, 5, 3)

        search_clicked = st.form_submit_button("Search")


# =========================================================
# RESULTS
# =========================================================

with right:
    st.subheader("My Recommendations")

    st.info(
        """
        **How to read the match score:**

        - **7.0–10.0** → Excellent match — You'll love this one!  
        - **5.0–6.9** → Good match, give it a try  
        - **3.0–4.9** → Quite different, but some similar notes
        """
    )

    direct_matches_box = st.container()

    if not search_clicked:
        st.info("Click **Search** to run recommendations.")
    else:
        # ---- Build query notes ----
        raw = split_notes(notes_text)
        query_notes = set(normalize_note(n) for n in raw)

        # ---- Direct hits in Chogan inspirations ----
        direct_hits = pd.DataFrame()
        if mode == "By perfume name" and perfume_name.strip():
            q = perfume_name.strip().lower()
            if "Inspiration" in chogan.columns:
                direct_hits = chogan[chogan["Inspiration"].fillna("").str.lower().str.contains(q, na=False)]
            else:
                direct_hits = chogan.iloc[0:0]

            # Optional brand narrow (simple contains)
            if brand_name.strip() and len(direct_hits) > 1 and "Inspiration" in direct_hits.columns:
                bq = brand_name.strip().lower()
                direct_hits = direct_hits[
                    direct_hits["Inspiration"].fillna("").str.lower().str.contains(bq, na=False)
                ]

        # ---- NEW: If searching by perfume name, ALSO seed from external sheet ----
        external_hit = pd.DataFrame()
        if mode == "By perfume name" and perfume_name.strip() and not external.empty:
            qn = norm_text(perfume_name)
            qb = norm_text(brand_name) if brand_name.strip() else ""

            m = external["Perfume"].fillna("").apply(norm_text).str.contains(qn, na=False)
            external_hit = external[m]

            if qb and len(external_hit) > 1:
                mb = external_hit["Brand"].fillna("").apply(norm_text).str.contains(qb, na=False)
                external_hit = external_hit[mb]

            if len(external_hit) > 0 and not query_notes:
                ext = external_hit.iloc[0]
                query_notes = set(
                    normalize_note(n)
                    for n in (
                        split_notes(ext.get("Top Notes", "")) +
                        split_notes(ext.get("Heart Notes", "")) +
                        split_notes(ext.get("Base Notes", "")) +
                        split_notes(ext.get("All Notes", ""))
                    )
                    if n
                )
                st.info(f"Using saved external notes for: {ext.get('Perfume','')} ({ext.get('Brand','')})")

        # ---- If direct hit exists and query_notes still empty, seed from direct hit ----
        with direct_matches_box:
            if len(direct_hits) > 0:
                st.success(f"Direct match found in Chogan inspirations ({len(direct_hits)})")
                for rank, (_, hit) in enumerate(direct_hits.head(top_n).iterrows(), start=1):
                    ref = hit.get("Perfume reference", "")
                    st.markdown(f"### ✅ Direct match #{rank} — {ref}")
                    st.write(f"Inspiration: *{hit.get('Inspiration','')}*")
                    st.write(f"Top: {hit.get('Top Notes','')}")
                    st.write(f"Heart: {hit.get('Heart Notes','')}")
                    st.write(f"Base: {hit.get('Base Notes','')}")
                    st.divider()

        if len(direct_hits) > 0 and not query_notes:
            seed = direct_hits.iloc[0]
            query_notes = set(
                normalize_note(n)
                for n in (
                    split_notes(seed.get("Top Notes", "")) +
                    split_notes(seed.get("Heart Notes", "")) +
                    split_notes(seed.get("Base Notes", ""))
                )
                if n
            )
            st.info("Using the direct match notes to generate secondary recommendations.")

        # ---- If still no notes, stop gracefully ----
        if not query_notes:
            st.warning("Please enter some notes, or search a perfume name that exists in your external sheet.")
        else:
            # ---- Apply filters (optional; safe) ----
            filtered = chogan.copy()

            if family_filter.strip() and "Olfactory Family" in filtered.columns:
                fq = family_filter.strip().lower()
                filtered = filtered[filtered["Olfactory Family"].fillna("").str.lower().str.contains(fq, na=False)]

            if "Gender" in filtered.columns:
                g = filtered["Gender"].fillna("").astype(str).str.strip().str.upper()
                if gender_choice == "Women (F)":
                    filtered = filtered[g == "F"]
                elif gender_choice == "Men (M)":
                    filtered = filtered[g == "M"]
                elif gender_choice == "Unisex (U)":
                    filtered = filtered[g == "U"]

            # ---- Score ----
            results = []
            for _, row in filtered.iterrows():
                score = score_perfume(query_notes, row)

                # (Optional) name-based boost without hiding everything
                if mode == "By perfume name" and perfume_name.strip():
                    sim = name_similarity(perfume_name, row.get("Inspiration", ""))
                    if sim > 0.85:
                        score = min(score + 1.2, 10.0)
                    elif sim > 0.70:
                        score = min(score + 0.6, 10.0)

                results.append((score, row))

            results.sort(key=lambda x: x[0], reverse=True)

            # ---- Render recommendations (allow secondary recs even if perfect match exists) ----
            shown = 0
            for score, row in results:
                if score < MIN_SCORE_TO_SHOW:
                    continue

                # If we have direct hits, don't show the SAME inspiration again
                if len(direct_hits) > 0 and perfume_name.strip():
                    # skip only if it's essentially the same inspiration
                    if name_similarity(perfume_name, row.get("Inspiration", "")) > 0.92:
                        continue

                shown += 1
                ref = row.get("Perfume reference", "")
                st.markdown(f"### #{shown} — {ref}")
                st.write(f"Inspiration: *{row.get('Inspiration','')}*")
                st.write(f"**Match score:** {score:.2f}/10")
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


# =========================================================
# ADD EXTERNAL PERFUME
# =========================================================

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
        st.error("Perfume name required")
    elif external_ws is None:
        st.error("Google Sheets is not connected.")
    else:
        row = {
            "Perfume": new_perfume.strip(),
            "Brand": new_brand.strip(),
            "Gender": new_gender.strip(),
            "Top Notes": new_top.strip(),
            "Heart Notes": new_heart.strip(),
            "Base Notes": new_base.strip(),
            "All Notes": new_all.strip(),
            "Olfactory Family": new_family.strip(),
        }

        upsert_external_to_sheets(external_ws, row)
        st.success("Saved")
        st.rerun()

with st.expander("View saved external perfumes"):
    st.dataframe(external.tail(50))
