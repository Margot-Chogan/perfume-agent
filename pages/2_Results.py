import streamlit as st

st.set_page_config(page_title="Find your Chogan Perfume", layout="centered")

# =========================================================
# SESSION DEFAULTS
# =========================================================
if "last_query" not in st.session_state:
    st.session_state.last_query = {}

# =========================================================
# PAGE STYLES
# =========================================================
st.markdown(
    """
    <style>
    .search-card{
        border:1px solid rgba(148,163,184,0.35);
        border-radius:16px;
        padding:16px;
        background: rgba(255,255,255,0.03);
    }
    .block-container{
        max-width: 820px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# PAGE
# =========================================================
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
            st.switch_page("pages/2_Results.py")

    with c2:
        if st.button("Reset", use_container_width=True):
            st.session_state.last_query = {}
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

st.info("Use the page tabs on the left/top to add new perfumes to the database.")
