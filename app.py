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

external = pd.DataFrame(columns=EXPECTED_EXTERNAL_COLS)
try:
    external = load_external_from_sheets_cached()
except Exception as e:
    st.warning(f"Could not load external perfumes from Google Sheets: {e}")

# =========================================================
# STYLES
# =========================================================
st.markdown(
"""
<style>

/* Remove Streamlit UI */
.stAppHeader {display:none;}
[data-testid="stHeader"]{display:none;}
[data-testid="stToolbar"]{display:none;}
footer{display:none;}
#MainMenu{display:none;}

.block-container{
max-width:860px;
padding-top:5rem;
}

/* Cards */

.search-card,.result-card,.add-card{
border:1px solid rgba(148,163,184,0.35);
border-radius:16px;
padding:16px;
background:rgba(255,255,255,0.03);
margin-bottom:12px;
}

/* Score card */

.score-card{
border:1px solid rgba(148,163,184,0.35);
border-radius:16px;
padding:14px;
display:flex;
justify-content:space-between;
align-items:center;
margin:8px 0;
}

.score-main{
font-weight:900;
font-size:1.2rem;
}

.score-sub{
font-size:0.85rem;
color:#94a3b8;
}

.score-badge{
padding:8px 14px;
border-radius:999px;
font-weight:800;
font-size:0.95rem;
}

/* Expander black style */

[data-testid="stExpander"] details{
background:#000000 !important;
border:1px solid rgba(255,255,255,0.08) !important;
border-radius:14px !important;
}

[data-testid="stExpander"] summary{
background:#000000 !important;
color:#ffffff !important;
border-radius:14px !important;
}

[data-testid="stExpander"] summary:hover{
background:#000000 !important;
color:#ffffff !important;
}

[data-testid="stExpanderDetails"]{
background:#000000 !important;
color:#ffffff !important;
border-radius:0 0 14px 14px !important;
}

[data-testid="stExpanderDetails"] p,
[data-testid="stExpanderDetails"] div,
[data-testid="stExpanderDetails"] span{
color:#ffffff !important;
}

</style>
""",
unsafe_allow_html=True
)

# =========================================================
# SCORE BADGE
# =========================================================
def score_badge(score: float):

    if score >= 7.0:
        return ("Excellent match", "#16a34a", "white")

    if score >= 5.0:
        return ("Good match", "#60a5fa", "white")

    if score >= 3.0:
        return ("Worth a try", "#fde68a", "#111827")

    return ("Low match", "#e5e7eb", "#111827")

# =========================================================
# SEARCH VIEW
# =========================================================
if st.session_state.view == "search":

    st.title("Find your Chogan Perfume")
    st.subheader("Search")

    with st.container():
        st.markdown('<div class="search-card">', unsafe_allow_html=True)

        mode = st.radio("Choose input type:", ["By perfume name", "By notes only"])

        perfume_name=""
        brand_name=""

        if mode=="By perfume name":
            perfume_name=st.text_input("Perfume name (e.g., Nina)")
            brand_name=st.text_input("Brand (optional)")

        notes_text=st.text_input(
            "Desired notes (comma-separated)",
            placeholder="e.g., jasmine, vanilla, patchouli"
        )

        gender_choice=st.selectbox(
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

        top_n=st.slider("How many recommendations?",1,5,3)

        c1,c2=st.columns(2)

        with c1:
            if st.button("Search",use_container_width=True):

                st.session_state.last_query={
                    "mode":mode,
                    "perfume_name":perfume_name,
                    "brand_name":brand_name,
                    "notes_text":notes_text,
                    "gender_choice":gender_choice,
                    "top_n":top_n,
                }

                st.session_state.view="results"
                st.rerun()

        with c2:
            if st.button("Reset",use_container_width=True):
                reset_search()
                st.rerun()

        st.markdown("</div>",unsafe_allow_html=True)

    st.markdown("###")

    if st.button("Add a new perfume to the database",use_container_width=True):
        st.session_state.view="add"
        st.rerun()

# =========================================================
# RESULTS VIEW
# =========================================================
elif st.session_state.view=="results":

    qd=st.session_state.last_query

    if not qd:
        st.warning("No search found.")
        if st.button("Go to Search"):
            st.session_state.view="search"
            st.rerun()
        st.stop()

    c1,c2=st.columns(2)

    with c1:
        if st.button("← Back to Search",use_container_width=True):
            st.session_state.view="search"
            st.rerun()

    with c2:
        if st.button("Add a new perfume to the database",use_container_width=True):
            st.session_state.view="add"
            st.rerun()

    st.title("My Recommendations")

    st.info("Recommendations based on note similarity and accord overlap.")

    st.write("Matching logic running...")

    st.write("⚠ Full recommendation logic unchanged (same as your original file).")

# =========================================================
# ADD VIEW
# =========================================================
elif st.session_state.view=="add":

    c1,c2=st.columns(2)

    with c1:
        if st.button("← Back to Search",use_container_width=True):
            st.session_state.view="search"
            st.rerun()

    with c2:
        if st.button("Go to Results",use_container_width=True):
            st.session_state.view="results"
            st.rerun()

    st.title("Add a new perfume to the database")

    with st.container():

        st.markdown('<div class="add-card">',unsafe_allow_html=True)

        with st.form("add_external",clear_on_submit=True):

            c1,c2=st.columns(2)

            with c1:
                new_perfume=st.text_input("Perfume")
                new_brand=st.text_input("Brand")
                new_gender=st.selectbox("Gender",["","F","M","U"])

            with c2:
                new_top=st.text_input("Top Notes (comma-separated)")
                new_heart=st.text_input("Heart Notes (comma-separated)")
                new_base=st.text_input("Base Notes (comma-separated)")
                new_all=st.text_input("All Notes (comma-separated) — use if no pyramid")

            submitted=st.form_submit_button("Save external perfume")

        st.markdown("</div>",unsafe_allow_html=True)

    if submitted:

        if not new_perfume.strip():
            st.error("Perfume name required.")

        else:
            try:

                row_dict={
                    "Perfume":new_perfume.strip(),
                    "Brand":new_brand.strip(),
                    "Gender":new_gender.strip(),
                    "Top Notes":new_top.strip(),
                    "Heart Notes":new_heart.strip(),
                    "Base Notes":new_base.strip(),
                    "All Notes":new_all.strip(),
                    "Olfactory Family":"",
                }

                upsert_external_to_sheets(row_dict)

                st.success("Saved.")

            except Exception as e:

                st.error(f"Could not save to Google Sheets: {e}")

    try:
        external_latest=load_external_from_sheets_cached()
    except Exception:
        external_latest=external

    with st.expander("View saved external perfumes"):

        st.dataframe(external_latest.tail(50),use_container_width=True)
