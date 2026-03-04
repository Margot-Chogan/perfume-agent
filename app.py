import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Find your Chogan Perfume", layout="wide")

# ---------- Helpers ----------
def split_notes(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    # split on commas, semicolons, slashes
    parts = re.split(r"[,/;]+", str(x))
    return [p.strip().lower() for p in parts if p.strip()]

# Small starter synonym map (you will expand this over time)
SYNONYMS = {
    "cedarwood": "cedar",
    "woody notes": "woody",
    "woods": "woody",
    "musk": "musk",
    "white musk": "musk",
    "orange blossom": "orange blossom",
    "neroli": "neroli",
}

EXPAND_KEYWORDS = {
    # Broad categories -> likely notes in databases
    "wood": ["woody", "woods", "cedar", "sandalwood", "vetiver", "patchouli", "guaiac wood", "cashmeran", "oakmoss"],
    "woody": ["woody", "woods", "cedar", "sandalwood", "vetiver", "patchouli", "cashmeran", "oakmoss"],
    "berry": ["berries", "red berries", "wild berries", "blackcurrant", "currant", "raspberry", "strawberry"],
    "citrus": ["bergamot", "lemon", "lime", "orange", "grapefruit", "mandarin"],
    "floral": ["rose", "jasmine", "orange blossom", "ylang-ylang", "tuberose", "iris", "violet", "peony", "lavender"],
    "vanilla": ["vanilla", "tonka bean", "benzoin"],
    "amber": ["amber", "ambergris", "labdanum", "benzoin"],
    "musk": ["musk", "white musk", "ambergris"],
}

def normalize_note(n):
    n = n.strip().lower()
    return SYNONYMS.get(n, n)

def expand_query_notes(raw_notes_list):
    expanded = set()

    for n in raw_notes_list:
        n = normalize_note(n)
        expanded.add(n)

        # if it's a broad keyword, expand it
        if n in EXPAND_KEYWORDS:
            for extra in EXPAND_KEYWORDS[n]:
                expanded.add(normalize_note(extra))

    return expanded

def notes_set(row):
    top = split_notes(row.get("Top Notes", ""))
    heart = split_notes(row.get("Heart Notes", ""))
    base = split_notes(row.get("Base Notes", ""))
    alln = [normalize_note(n) for n in (top + heart + base)]
    return set([n for n in alln if n])

def weighted_score(query_notes, row):

    # collect notes from each layer
    top = set(normalize_note(n) for n in split_notes(row.get("Top Notes", "")))
    heart = set(normalize_note(n) for n in split_notes(row.get("Heart Notes", "")))
    base = set(normalize_note(n) for n in split_notes(row.get("Base Notes", "")))
    all_notes = set(normalize_note(n) for n in split_notes(row.get("All Notes", "")))

    score = 0.0
    matched = set()

    for n in query_notes:

        if n in top:
            score += 1.0
            matched.add(n)

        elif n in heart:
            score += 1.2
            matched.add(n)

        elif n in base:
            score += 1.3
            matched.add(n)

        elif n in all_notes:
            score += 1.1
            matched.add(n)

    # bonus if all notes appear somewhere
    all_combined = top | heart | base | all_notes

    if query_notes and query_notes.issubset(all_combined):
        score += 2.0

    return score, matched, all_combined

# ---------- Load data ----------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

try:
    chogan = load_csv("chogan_catalog.csv")
except Exception as e:
    st.error("Could not load chogan_catalog.csv. Make sure it is in the repo.")
    st.stop()

try:
    external = load_csv("external_perfumes.csv")
except Exception:
    external = pd.DataFrame(columns=["Perfume", "Brand", "Top Notes", "Heart Notes", "Base Notes", "All Notes", "Olfactory Family"])

# ---------- UI ----------
st.title("Find your Chogan Perfume")

left, right = st.columns([1, 2])

with left:
    st.subheader("Search Mode")
    mode = st.radio("Choose input type:", ["By non-Chogan perfume name", "By notes only"])

    perfume_name = ""
    if mode == "By non-Chogan perfume name":
        perfume_name = st.text_input("Perfume name (e.g., Nina)")
        brand_name = st.text_input("Brand (optional, e.g., Nina Ricci)")
    else:
        brand_name = ""

    notes_text = st.text_input("Desired notes (comma-separated)", placeholder="e.g., jasmine, lavender, woody")

    st.subheader("Filters (optional)")
    family_filter = st.text_input("Olfactory family contains", placeholder="e.g., floral, oriental, woody")
    gender_choice = st.selectbox(
    "Gender preference",
    ["Any", "Women (F)", "Men (M)", "Unisex (U)", "Women or Unisex (F/U)", "Men or Unisex (M/U)"]
)

    top_n = st.slider("How many results?", 1, 3, 5)

with right:
    st.subheader("Recommendations")

    # Build query notes
    raw = split_notes(notes_text)
    query_notes = expand_query_notes(raw)

    # If perfume name mode, try to find saved external perfume and use its notes
    used_external = None
    if mode == "By non-Chogan perfume name" and perfume_name.strip():
        mask = external["Perfume"].fillna("").str.lower().str.contains(perfume_name.strip().lower())
        if brand_name.strip():
            mask = mask & external["Brand"].fillna("").str.lower().str.contains(brand_name.strip().lower())
        matches = external[mask]

        if len(matches) > 0:
            used_external = matches.iloc[0].to_dict()
            ext_notes = set()
            for col in ["Top Notes", "Heart Notes", "Base Notes", "All Notes"]:
                ext_notes |= set(normalize_note(n) for n in split_notes(used_external.get(col, "")))
            query_notes |= ext_notes
            st.info(f"Using saved notes for: {used_external.get('Perfume','')} ({used_external.get('Brand','')})")
        else:
            st.warning("No saved notes found for that perfume. Add it below (manual entry) to reuse next time.")

    # Apply filters
    filtered = chogan.copy()

    if family_filter.strip():
        filtered = filtered[
            filtered["Olfactory Family"].fillna("").str.lower().str.contains(family_filter.strip().lower())
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
        # Any = no filter

    # Score and rank
    results = []
    for _, row in filtered.iterrows():
        sc, matched, alln = weighted_score(query_notes, row)
        results.append((sc, matched, row))

    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_n]

    if not query_notes:
        st.write("Enter notes (or select a saved external perfume) to get recommendations.")
    else:
        for rank, (sc, matched, row) in enumerate(top_results, start=1):
            # Try multiple possible column names for the reference
            ref = row.get("Perfume ref.") or row.get("Reference") or row.get("Code") or row.get("ID") or ""
            insp = row.get("Inspiration", "")
            fam = row.get("Olfactory Family", "")
            top = row.get("Top Notes", "")
            heart = row.get("Heart Notes", "")
            base = row.get("Base Notes", "")

            st.markdown(f"### #{rank} — **{ref}**")
            st.write(f"Inspiration: *{insp}*")
            st.write(f"Family: *{fam}*")
            st.write(f"**Match score:** {sc:.2f}")
            st.write(f"**Matched notes:** {', '.join(sorted(matched)) if matched else 'None'}")
            st.write(f"Top: {top}")
            st.write(f"Heart: {heart}")
            st.write(f"Base: {base}")
            st.divider()

st.subheader("Add / Update an External (non-Chogan) Perfume (manual entry)")
with st.form("add_external"):
    c1, c2 = st.columns(2)
    with c1:
        new_perfume = st.text_input("Perfume")
        new_brand = st.text_input("Brand")
        new_family = st.text_input("Olfactory Family (optional)")
    with c2:
        new_top = st.text_input("Top Notes (comma-separated)")
        new_heart = st.text_input("Heart Notes (comma-separated)")
        new_base = st.text_input("Base Notes (comma-separated)")
        new_all = st.text_input("All Notes (comma-separated) — use if no pyramid")

    submitted = st.form_submit_button("Save external perfume")

if submitted:
    if not new_perfume.strip():
        st.error("Perfume name is required.")
    else:
        # Upsert by Perfume + Brand
        if "Perfume" not in external.columns:
            external = pd.DataFrame(columns=["Perfume", "Brand", "Top Notes", "Heart Notes", "Base Notes", "Olfactory Family"])

        key_mask = (external["Perfume"].fillna("").str.lower() == new_perfume.strip().lower()) & \
                   (external["Brand"].fillna("").str.lower() == new_brand.strip().lower())

        new_row = {
            "Perfume": new_perfume.strip(),
            "Brand": new_brand.strip(),
            "Top Notes": new_top.strip(),
            "Heart Notes": new_heart.strip(),
            "Base Notes": new_base.strip(),
            "All Notes": new_all.strip(),
            "Olfactory Family": new_family.strip(),
        }

        if key_mask.any():
            external.loc[key_mask, :] = pd.DataFrame([new_row]).iloc[0]
        else:
            external = pd.concat([external, pd.DataFrame([new_row])], ignore_index=True)

        external.to_csv("external_perfumes.csv", index=False)
        st.success("Saved! (Note: on Streamlit Cloud, you’ll want to store this in Google Sheets or a small database—see next step.)")
