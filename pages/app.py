import streamlit as st
import pandas as pd
import json
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Add perfume", layout="centered")

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


st.title("Add a new perfume to the database")

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
            st.rerun()
        except Exception as e:
            st.error(f"Could not save to Google Sheets: {e}")

try:
    external = load_external_from_sheets_cached()
except Exception as e:
    st.warning(f"Could not load external perfumes from Google Sheets: {e}")
    external = pd.DataFrame(columns=EXPECTED_EXTERNAL_COLS)

with st.expander("View saved external perfumes"):
    st.dataframe(external.tail(50), use_container_width=True)
