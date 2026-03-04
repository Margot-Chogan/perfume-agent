import streamlit as st
import pandas as pd
import re
import json
import gspread
from google.oauth2.service_account import Credentials

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

@st.cache_resource
def get_gs_client():
    creds_info = json.loads(st.secrets["gcp_service_account"]["raw_json"])
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
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

def weighted_score(query_notes, row, query_top=None, query_heart=None, query_base=None):

    top = set(normalize_note(n) for n in split_notes(row.get("Top Notes", "")))
    heart = set(normalize_note(n) for n in split_notes(row.get("Heart Notes", "")))
    base = set(normalize_note(n) for n in split_notes(row.get("Base Notes", "")))

    score = 0.0
    matched = set()

    # Pyramid-aware scoring
    if query_top:
        for n in query_top:
            if n in top:
                score += 2.0
                matched.add(n)
            elif n in heart:
                score += 1.2
                matched.add(n)
            elif n in base:
                score += 0.8
                matched.add(n)

    if query_heart:
        for n in query_heart:
            if n in heart:
                score += 1.6
                matched.add(n)
            elif n in top:
                score += 1.2
                matched.add(n)
            elif n in base:
                score += 1.0
                matched.add(n)

    if query_base:
        for n in query_base:
            if n in base:
                score += 1.4
                matched.add(n)
            elif n in heart:
                score += 1.1
                matched.add(n)

    # fallback if pyramid not used
    if not (query_top or query_heart or query_base):
        for n in query_notes:
            if n in top:
                score += 1.6
                matched.add(n)
            elif n in heart:
                score += 1.2
                matched.add(n)
            elif n in base:
                score += 1.0
                matched.add(n)

    return score, matched, top | heart | base

def get_external_worksheet():
    gc = get_gs_client()
    sheet_id = st.secrets["external_sheet"]["spreadsheet_id"]
    ws_name = st.secrets["external_sheet"]["worksheet_name"]
    return gc.open_by_key(sheet_id).worksheet(ws_name)

def ensure_external_headers(ws):
    """Ensure row 1 headers match EXPECTED_EXTERNAL_COLS."""
    headers = ws.row_values(1)
    if headers != EXPECTED_EXTERNAL_COLS:
        ws.clear()
        ws.append_row(EXPECTED_EXTERNAL_COLS)

def load_external_from_sheets():
    gc = get_gs_client()
    sheet_id = st.secrets["external_sheet"]["spreadsheet_id"]
    ws_name = st.secrets["external_sheet"]["worksheet_name"]

    ws = gc.open_by_key(sheet_id).worksheet(ws_name)
    records = ws.get_all_records()

    df = pd.DataFrame(records)
    for col in EXPECTED_EXTERNAL_COLS:
        if col not in df.columns:
            df[col] = ""

    return df[EXPECTED_EXTERNAL_COLS], ws

def standardize_external_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ["perfume", "perfume name", "name", "fragrance", "parfum"]:
            rename_map[c] = "Perfume"
        elif lc in ["brand", "house", "designer", "maker"]:
            rename_map[c] = "Brand"
        elif lc in ["top", "top notes", "head notes"]:
            rename_map[c] = "Top Notes"
        elif lc in ["heart", "middle", "middle notes", "mid notes"]:
            rename_map[c] = "Heart Notes"
        elif lc in ["base", "base notes"]:
            rename_map[c] = "Base Notes"
        elif lc in ["all notes", "notes", "notes (all)", "all"]:
            rename_map[c] = "All Notes"
        elif lc in ["olfactory family", "family", "accords"]:
            rename_map[c] = "Olfactory Family"
        elif lc in ["gender", "sex"]:
            rename_map[c] = "Gender"

    df = df.rename(columns=rename_map)

    for col in EXPECTED_EXTERNAL_COLS:
        if col not in df.columns:
            df[col] = ""

    return df[EXPECTED_EXTERNAL_COLS]

# ---------- Load data ----------
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
    
# ---------- UI ----------
st.title("Find your Chogan Perfume")

left, right = st.columns([1, 2])

with left:
    st.subheader("Search Mode")
    mode = st.radio("Choose input type:", ["By non-Chogan perfume name", "By notes only"])

    perfume_name = ""
    brand_name = ""

    if mode == "By non-Chogan perfume name":
        perfume_name = st.text_input("Perfume name (e.g., Nina)")
        brand_name = st.text_input("Brand (optional, e.g., Nina Ricci)")

    notes_text = st.text_input("Desired notes (comma-separated)", placeholder="e.g., jasmine, lavender, woody")

    st.subheader("Filters (optional)")
    family_filter = st.text_input("Olfactory family contains", placeholder="e.g., floral, oriental, woody")

    gender_choice = st.selectbox(
        "Gender preference",
        ["Any", "Women (F)", "Men (M)", "Unisex (U)", "Women or Unisex (F/U)", "Men or Unisex (M/U)"],
    )

    top_n = st.slider("How many recommendations?", 1, 5, 3)

with right:
    st.subheader("Recommendations")

    with st.expander("How to read the match score (out of 10)"):
        st.markdown(
        """
    - **9.0–10.0** → Excellent match  
    - **7.0–8.9** → Very good match  
    - **5.0–6.9** → Good match  
    - **3.0–4.9** → Possible match  
    - **0.0–2.9** → Weak match
        """
        )

    # Build query notes from typed notes
    raw = split_notes(notes_text)
    query_notes = expand_query_notes(raw)
    
    # Initialize pyramid query sets
    query_top = set()
    query_heart = set()
    query_base = set()

    # If searching by non-Chogan perfume name, pull notes from external DB
    if mode == "By non-Chogan perfume name" and perfume_name.strip():
        mask = external["Perfume"].fillna("").str.lower().str.contains(perfume_name.strip().lower(), na=False)
        matches = external[mask]

        # Only narrow by brand if multiple matches
        if brand_name.strip() and len(matches) > 1:
            bmask = matches["Brand"].fillna("").str.lower().str.contains(brand_name.strip().lower(), na=False)
            matches = matches[bmask]

        if len(matches) > 0:
            used_external = matches.iloc[0].to_dict()

            query_top = set(normalize_note(n) for n in split_notes(used_external.get("Top Notes", "")))
            query_heart = set(normalize_note(n) for n in split_notes(used_external.get("Heart Notes", "")))
            query_base = set(normalize_note(n) for n in split_notes(used_external.get("Base Notes", "")))

            ext_notes = query_top | query_heart | query_base

            # fallback if pyramid not specified
            if not ext_notes:
                ext_notes = set(normalize_note(n) for n in split_notes(used_external.get("All Notes", "")))

            query_notes |= ext_notes

            st.info(f"Using saved notes for: {used_external.get('Perfume','')} ({used_external.get('Brand','')})")
        else:
            st.warning("No saved notes found for that perfume. Add it below (manual entry) to reuse next time.")

    # Apply filters to Chogan catalog
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

    # Score and rank
    if not query_notes:
        st.write("Enter notes (or select a saved external perfume) to get recommendations.")
    else:
        results = []
        for _, row in filtered.iterrows():
            sc, matched, _ = weighted_score(query_notes, row, query_top, query_heart, query_base)
            results.append((sc, matched, row))

        results.sort(key=lambda x: x[0], reverse=True)

        # Keep searching down the list until we collect up to top_n non-zero results
        non_zero = [r for r in results if r[0] > 0][:top_n]

        # Compute maximum possible score for display
        max_score = len(query_notes) * 1.6 + 2
        if not non_zero:
            st.warning(
                "No close matches found. Try more specific notes (e.g., 'cedar', 'blackcurrant', 'vanilla') or remove one note."
            )
        else:
            for rank, (sc, matched, row) in enumerate(non_zero, start=1):
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
                score_10 = (sc / max_score) * 10
                st.write(f"**Match score:** {score_10:.2f} / 10")
                st.write(f"**Matched notes:** {', '.join(sorted(matched)) if matched else 'None'}")
                st.write(f"Top: {top}")
                st.write(f"Heart: {heart}")
                st.write(f"Base: {base}")
                st.divider()

# ---------- Add / Update External Perfume ----------
st.subheader("Add / Update an External (non-Chogan) Perfume (manual entry)")

with st.form("add_external"):
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
