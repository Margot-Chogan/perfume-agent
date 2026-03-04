import streamlit as st
import pandas as pd
import re
import json
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher

# ---------- External perfumes columns ----------
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

# ---------- Google Sheets ----------
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
    # always re-open worksheet fresh to avoid stale handles
    ws = get_external_worksheet()
    ensure_external_headers(ws)

    records = ws.get_all_records()

    key_perfume = row_dict.get("Perfume", "").strip().lower()
    key_brand = row_dict.get("Brand", "").strip().lower()

    target_row = None
    for i, r in enumerate(records, start=2):  # data starts at row 2
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

# ---------- Helpers ----------
def find_chogan_direct_matches(chogan_df: pd.DataFrame, perfume_query: str) -> pd.DataFrame:
    q = perfume_query.strip().lower()
    if not q or "Inspiration" not in chogan_df.columns:
        return chogan_df.iloc[0:0]
    return chogan_df[chogan_df["Inspiration"].fillna("").str.lower().str.contains(q, na=False)]

def split_notes(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    parts = re.split(r"[,/;]+", str(x))
    return [p.strip().lower() for p in parts if p.strip()]

SYNONYMS = {
    "cedarwood": "cedar",
    "woody notes": "woody",
    "woods": "woody",
    "white musk": "musk",
}

EXPAND_KEYWORDS = {
    "wood": ["woody", "woods", "cedar", "sandalwood", "vetiver", "patchouli", "guaiac wood", "cashmeran", "oakmoss"],
    "woody": ["woody", "woods", "cedar", "sandalwood", "vetiver", "patchouli", "cashmeran", "oakmoss"],
    "berry": ["berries", "red berries", "wild berries", "blackcurrant", "currant", "raspberry", "strawberry"],
    "citrus": ["bergamot", "lemon", "lime", "orange", "grapefruit", "mandarin"],
    "floral": ["rose", "jasmine", "orange blossom", "ylang-ylang", "tuberose", "iris", "violet", "peony", "lavender"],
    "vanilla": ["vanilla", "tonka bean", "benzoin"],
    "amber": ["amber", "ambergris", "labdanum", "benzoin"],
    "musk": ["musk", "ambergris"],
}

# -------- Accord / pillar detection (simple + tweakable) --------
PILLARS = {
    "fruity": {
        "pear","raspberry","strawberry","lychee","blackcurrant","currant","peach","plum","apple",
        "pineapple","mango","passionfruit","berry","berries","orange","mandarin","tangerine","bergamot","lemon","citrus"
    },
    "floral": {
        "rose","black rose","jasmine","jasmine sambac","orange blossom","neroli","peony","datura",
        "tuberose","ylang-ylang","iris","violet","orchid","vanilla orchid","lily"
    },
    "gourmand": {
        "vanilla","praline","caramel","toffee","chocolate","cocoa","coffee","honey","sugar","candy",
        "marshmallow","tonka","tonka bean","benzoin","almond"
    },
    "woody": {
        "cedar","cedarwood","sandalwood","vetiver","patchouli","moss","oakmoss","wood","woods","woody","papyrus"
    },
    "musky": {"musk","white musk","ambroxan","ambergris","ambrox"},
    "resinous": {"incense","labdanum","amber","resin","myrrh","opoponax"},
    "spicy": {"pepper","pink pepper","cinnamon","cardamom","clove","saffron"},
    "sweet": {"vanilla","praline","caramel","honey","sugar","tonka","benzoin"},
}

# notes -> which pillars are present
def detect_pillars(notes_set: set[str]) -> set[str]:
    found = set()
    blob = " | ".join(notes_set)  # one string
    for pillar, kws in PILLARS.items():
        for kw in kws:
            if kw in blob:
                found.add(pillar)
                break
    return found

def normalize_note(n: str) -> str:
    n = str(n).strip().lower()
    return SYNONYMS.get(n, n)

def expand_query_notes(raw_notes_list):
    expanded = set()
    for n in raw_notes_list:
        n = normalize_note(n)
        if not n:
            continue
        expanded.add(n)
        if n in EXPAND_KEYWORDS:
            for extra in EXPAND_KEYWORDS[n]:
                expanded.add(normalize_note(extra))
    return expanded

def compute_max_score(query_top, query_heart, query_base, query_notes_base, used_pyramid: bool) -> float:
    """
    used_pyramid=True  -> denominator based on the pyramid sets (Top/Heart/Base)
    used_pyramid=False -> denominator based on the notes-only query (query_notes_base)
    """
    if used_pyramid:
        return (
            2.0 * len(query_top) +
            1.6 * len(query_heart) +
            1.4 * len(query_base)
        )
    return 1.6 * len(query_notes_base)

def weighted_score(query_notes_match, row, query_top=None, query_heart=None, query_base=None, query_notes_base=None):
    top = set(normalize_note(n) for n in split_notes(row.get("Top Notes", "")))
    heart = set(normalize_note(n) for n in split_notes(row.get("Heart Notes", "")))
    base = set(normalize_note(n) for n in split_notes(row.get("Base Notes", "")))
    all_notes = top | heart | base

    score = 0.0
    matched = set()

    # Pyramid-aware scoring
    if query_top:
        for n in query_top:
            if n in top:
                score += 2.0; matched.add(n)
            elif n in heart:
                score += 1.2; matched.add(n)
            elif n in base:
                score += 0.8; matched.add(n)

    if query_heart:
        for n in query_heart:
            if n in heart:
                score += 1.6; matched.add(n)
            elif n in top:
                score += 1.2; matched.add(n)
            elif n in base:
                score += 1.0; matched.add(n)

    if query_base:
        for n in query_base:
            if n in base:
                score += 1.4; matched.add(n)
            elif n in heart:
                score += 1.1; matched.add(n)

    # fallback if pyramid not used
    if not (query_top or query_heart or query_base):
        for n in query_notes_match:
            if n in top:
                score += 1.6; matched.add(n)
            elif n in heart:
                score += 1.2; matched.add(n)
            elif n in base:
                score += 1.0; matched.add(n)

    # Perfect match boost (based on unexpanded notes)
    if query_notes_base:
        all_notes = top | heart | base
        base_hit = len(set(query_notes_base) & all_notes)
        base_total = len(set(query_notes_base))
        if base_total > 0:
            coverage = base_hit / base_total
            if coverage == 1.0:
                score += 2.0
            elif coverage >= 0.8:
                score += 1.0

    # -------- Accord / pillar bonus --------
    perfume_pillars = detect_pillars(all_notes)

    if query_notes_base:
        query_pillars = detect_pillars(set(query_notes_base))
        overlap = perfume_pillars & query_pillars

        # Strong pillars that usually define "same vibe"
        if "patchouli" in (top | heart | base) and "patchouli" in set(query_notes_base):
            score += 0.6  # small extra anchor bonus

        # Generic pillar overlap bonus (tweakable)
        score += 0.6 * len(overlap)

        # Extra synergy bonus for common "designer DNA" combos
        if {"fruity", "woody"} <= overlap:     # fruity + patchouli/woody
            score += 0.8
        if {"fruity", "gourmand"} <= overlap:  # fruity gourmand (LNT style)
            score += 0.6

    return score, matched

# ---------- Load data ----------
@st.cache_data
def load_chogan_csv(path):
    return pd.read_csv(path)

try:
    chogan = load_chogan_csv("chogan_catalog.csv")
except Exception:
    st.error("Could not load chogan_catalog.csv. Make sure it is in your repo.")
    st.stop()

import traceback
try:
    external, external_ws = load_external_from_sheets()
except Exception:
    st.error("Could not load external perfumes from Google Sheets (full error below):")
    st.code(traceback.format_exc())
    external = pd.DataFrame(columns=EXPECTED_EXTERNAL_COLS)
    external_ws = None

# ---------- UI ----------
st.title("Find your Chogan Perfume")

left, right = st.columns([1, 2])

with left:
    st.subheader("Search Mode")

    # ✅ Search only happens when the button is clicked
    with st.form("search_form"):
        mode = st.radio("Choose input type:", ["By perfume name", "By notes only"])

        perfume_name = ""
        brand_name = ""

        if mode == "By perfume name":
            perfume_name = st.text_input("Perfume name (e.g., Nina)")
            brand_name = st.text_input("Brand (optional, e.g., Nina Ricci)")

        notes_text = st.text_input(
            "Desired notes (comma-separated)",
            placeholder="e.g., jasmine, lavender, woody",
        )

        st.subheader("Filters (optional)")
        family_filter = st.text_input(
            "Olfactory family contains",
            placeholder="e.g., floral, oriental, woody",
        )

        gender_choice = st.selectbox(
            "Gender preference",
            ["Any", "Women (F)", "Men (M)", "Unisex (U)", "Women or Unisex (F/U)", "Men or Unisex (M/U)"],
        )

        top_n = st.slider("How many recommendations?", 1, 5, 3)

        # ✅ The only trigger for searching
        search_clicked = st.form_submit_button("Search")

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

    # This is where direct hits will appear (RIGHT side, under the score info)
    direct_matches_box = st.container()

    if not search_clicked:
        st.info("Click **Search** to run recommendations.")
    else:
        # ✅ EVERYTHING that currently starts at your line 331 onward
        # (direct hits, building query notes, filtering, scoring, rendering)
        # must be indented under this else block.

        # Example: (keep your existing code, just indent it)
        # raw = split_notes(notes_text)
        # query_notes_base = ...
        # ...
        pass  # remove this after you paste your code under the else

        # ---- Build query notes from typed notes ----
        raw = split_notes(notes_text)
        query_notes_base = set(normalize_note(n) for n in raw)   # denominator
        query_notes_match = expand_query_notes(raw)              # matching
    
        query_top, query_heart, query_base = set(), set(), set()
        used_pyramid_query = False
        direct_hits = chogan.iloc[0:0]
    
        # ---- If searching by perfume name ----
        if mode == "By perfume name" and perfume_name.strip():
            # A) Direct hits in Chogan inspirations
            direct_hits = find_chogan_direct_matches(chogan, perfume_name)
    
            if brand_name.strip() and len(direct_hits) > 1:
                direct_hits = direct_hits[
                    direct_hits["Inspiration"].fillna("").str.lower().str.contains(brand_name.strip().lower(), na=False)
                ]
    
            # B) Pull notes from external DB (to fuel note-based recommendations)
            mask = external["Perfume"].fillna("").str.lower().str.contains(perfume_name.strip().lower(), na=False)
            matches = external[mask]
    
            if brand_name.strip() and len(matches) > 1:
                bmask = matches["Brand"].fillna("").str.lower().str.contains(brand_name.strip().lower(), na=False)
                matches = matches[bmask]
    
            if len(matches) > 0:
                used_external = matches.iloc[0].to_dict()
            
                query_top = set(normalize_note(n) for n in split_notes(used_external.get("Top Notes", "")))
                query_heart = set(normalize_note(n) for n in split_notes(used_external.get("Heart Notes", "")))
                query_base = set(normalize_note(n) for n in split_notes(used_external.get("Base Notes", "")))
            
                ext_notes = query_top | query_heart | query_base
            
                if ext_notes:
                    used_pyramid_query = True
                else:
                    # fallback if pyramid not provided
                    ext_notes = set(normalize_note(n) for n in split_notes(used_external.get("All Notes", "")))
                    used_pyramid_query = False
            
                query_notes_match |= ext_notes
                query_notes_base |= ext_notes
    
                st.info(f"Using saved notes for: {used_external.get('Perfume','')} ({used_external.get('Brand','')})")
            else:
                if len(direct_hits) == 0:
                    st.warning("No saved notes found and no direct Chogan inspiration match. Try a different name or add it below.")
    
        # ✅ Render direct hits ON THE RIGHT
        with direct_matches_box:
            if len(direct_hits) > 0:
                st.success(f"Direct match found in Chogan inspirations ({len(direct_hits)} result(s)).")
    
                for rank, (_, hit) in enumerate(direct_hits.head(top_n).iterrows(), start=1):
                    ref = (
                        hit.get("Perfume reference")
                        or hit.get("Perfume ref.")
                        or hit.get("Reference")
                        or hit.get("Code")
                        or hit.get("ID")
                        or ""
                    )
    
                    st.markdown(f"### ✅ Direct match #{rank} — **{ref}**")
                    st.write(f"Inspiration: *{hit.get('Inspiration','')}*")
                    st.write(f"Family: *{hit.get('Olfactory Family','')}*")
                    st.write(f"Top: {hit.get('Top Notes','')}")
                    st.write(f"Heart: {hit.get('Heart Notes','')}")
                    st.write(f"Base: {hit.get('Base Notes','')}")
                    st.divider()

    # ---- Apply filters to Chogan catalog ----
    filtered = chogan.copy()

    if family_filter.strip() and "Olfactory Family" in filtered.columns:
        filtered = filtered[
            filtered["Olfactory Family"].fillna("").str.lower().str.contains(family_filter.strip().lower(), na=False)
        ]

    if "Gender" in filtered.columns:
        g = filtered["Gender"].fillna("").astype(str).str.strip().str.upper()
        if gender_choice == "Women (F)":
            filtered = filtered[g == "F"]
        elif gender_choice == "Men (M)":
            filtered = filtered[g == "M"]
        elif gender_choice == "Unisex (U)":
            filtered = filtered[g == "U"]
        elif gender_choice == "Women or Unisex (F/U)":
            filtered = filtered[g.isin(["F", "U"])]
        elif gender_choice == "Men or Unisex (M/U)":
            filtered = filtered[g.isin(["M", "U"])]

    # ✅ Remove direct-hit inspirations from recommendations (prevents duplicate Coco Mademoiselle)
    if len(direct_hits) > 0 and "Inspiration" in filtered.columns:
        direct_insp = set(direct_hits["Inspiration"].fillna("").astype(str).str.lower())
        filtered = filtered[~filtered["Inspiration"].fillna("").astype(str).str.lower().isin(direct_insp)]

    # ---- Score & rank recommendations ----
    if not query_notes_base:
        st.write("Enter notes (or select a saved perfume) to get recommendations.")
    else:
        max_score = compute_max_score(query_top, query_heart, query_base, query_notes_base, used_pyramid_query)
        if max_score <= 0:
            st.write("Enter notes (or select a saved perfume) to get recommendations.")
        else:
            results = []
            for _, row in filtered.iterrows():
                sc, matched = weighted_score(
                    query_notes_match,
                    row,
                    query_top,
                    query_heart,
                    query_base,
                    query_notes_base=query_notes_base,
                )

                # Fuzzy inspiration name boost (makes name searches feel smarter)
                if mode == "By perfume name" and perfume_name.strip():
                    search = perfume_name.strip().lower()
                    insp_text = str(row.get("Inspiration", "")).lower()
                    similarity = SequenceMatcher(None, search, insp_text).ratio()
                    if similarity > 0.8:
                        sc += 5.0
                    elif similarity > 0.6:
                        sc += 3.0

                results.append((sc, matched, row))

            results.sort(key=lambda x: x[0], reverse=True)

            # Only show >= 3/10
            good_matches = [r for r in results if (r[0] / max_score) * 10 >= 3][:top_n]

            if not good_matches:
                st.warning(
                    "Sorry, we don't have a good match for the notes in the perfume you are looking for. "
                    "Would you like to try something else?"
                )
            else:
                for rank, (sc, matched, row) in enumerate(good_matches, start=1):
                    score_10 = min((sc / max_score) * 10, 10.0)

                    ref = (
                        row.get("Perfume reference")
                        or row.get("Perfume ref.")
                        or row.get("Reference")
                        or row.get("Code")
                        or row.get("ID")
                        or ""
                    )

                    insp = row.get("Inspiration", "")
                    fam = row.get("Olfactory Family", "")
                    top = row.get("Top Notes", "")
                    heart = row.get("Heart Notes", "")
                    base = row.get("Base Notes", "")

                    st.markdown(f"### #{rank} — **{ref}**")
                    st.write(f"Inspiration: *{insp}*")
                    st.write(f"Family: *{fam}*")
                    st.write(f"**Match score:** {score_10:.2f} / 10")
                    st.write(f"**Matched notes:** {', '.join(sorted(matched)) if matched else 'None'}")
                    st.write(f"Top: {top}")
                    st.write(f"Heart: {heart}")
                    st.write(f"Base: {base}")
                    st.divider()

# ---------- Add / Update External Perfume ----------
st.subheader("Add / Update an External Perfume (manual entry)")

with st.form("add_external", clear_on_submit=True):
    c1, c2 = st.columns(2)

    with c1:
        new_perfume = st.text_input("Perfume")
        new_brand = st.text_input("Brand")
        new_family = st.text_input("Olfactory Family (optional)")
        new_gender = st.selectbox("Gender (optional)", ["", "F", "M", "U"], help="F=Women, M=Men, U=Unisex")

    with c2:
        new_top = st.text_input("Top Notes (comma-separated)")
        new_heart = st.text_input("Heart Notes (comma-separated)")
        new_base = st.text_input("Base Notes (comma-separated)")
        new_all = st.text_input("All Notes (comma-separated) — use if no pyramid")

    submitted = st.form_submit_button("Save external perfume")

if submitted:
    if not new_perfume.strip():
        st.error("Perfume name is required.")
    elif external_ws is None:
        st.error("Google Sheets is not connected yet. Fix the error above first.")
    else:
        new_row = {
            "Perfume": new_perfume.strip(),
            "Brand": new_brand.strip(),
            "Gender": new_gender.strip(),
            "Top Notes": new_top.strip(),
            "Heart Notes": new_heart.strip(),
            "Base Notes": new_base.strip(),
            "All Notes": new_all.strip(),
            "Olfactory Family": new_family.strip(),
        }

        upsert_external_to_sheets(external_ws, new_row)
        st.success("Saved (updated if already existed).")
        st.rerun()

with st.expander("View saved external perfumes"):
    st.dataframe(external.tail(50))
