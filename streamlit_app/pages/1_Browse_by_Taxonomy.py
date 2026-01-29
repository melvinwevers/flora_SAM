"""
Browse Plants - navigate through taxonomy with color palettes
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils import data_loader, charts

st.set_page_config(page_title="Browse Plants", page_icon="🌿", layout="wide")

st.title("🌿 Browse Plants")

# Sidebar filters
st.sidebar.header("Filters")

# Family selection
families = data_loader.get_families()
family_options = ["All Families"] + families.index.tolist()
selected_family = st.sidebar.selectbox("Family", family_options)

# Genus selection (filtered by family)
if selected_family != "All Families":
    genera = data_loader.get_genera(family=selected_family)
    genus_options = ["All Genera"] + genera.index.tolist()
else:
    genera = data_loader.get_genera()
    genus_options = ["All Genera"] + genera.index.tolist()

selected_genus = st.sidebar.selectbox("Genus", genus_options)

# Text search
search_query = st.sidebar.text_input(
    "Search by name",
    "",
    placeholder="Plant ID, genus, species, or Dutch name..."
)

st.sidebar.markdown("---")
st.sidebar.header("Display Options")

# Color ranking toggle
color_ranking = st.sidebar.radio(
    "Color Ranking",
    options=["frequency", "perceptual", "saliency"],
    format_func=lambda x: {"frequency": "Frequency", "perceptual": "Perceptual", "saliency": "Salience"}[x]
)

# Number of colors
num_colors = st.sidebar.slider("Colors shown", 3, 5, 5)

# Items per page
items_per_page = st.sidebar.selectbox("Plants per page", [20, 40, 60], index=1)

# Get filtered plants
family_filter = None if selected_family == "All Families" else selected_family
genus_filter = None if selected_genus == "All Genera" else selected_genus

filtered_df = data_loader.get_plants_by_taxonomy(
    family=family_filter,
    genus=genus_filter
)

# Apply text search
if search_query:
    query_lower = search_query.lower()
    filtered_df = filtered_df[
        filtered_df['plant_id'].str.lower().str.contains(query_lower) |
        filtered_df['genus'].fillna('').str.lower().str.contains(query_lower) |
        filtered_df['species'].fillna('').str.lower().str.contains(query_lower) |
        filtered_df['Huidige Nederlandse naam'].fillna('').str.lower().str.contains(query_lower)
    ]

# Summary metrics
col1, col2, col3 = st.columns(3)
col1.metric("Plants Found", len(filtered_df))
if family_filter:
    col2.metric("Family", family_filter)
if genus_filter:
    col3.metric("Genus", genus_filter)

# Plants grid
if len(filtered_df) == 0:
    st.warning("No plants found with the selected filters.")
else:
    # Pagination
    total_pages = max(1, (len(filtered_df) + items_per_page - 1) // items_per_page)

    # Initialize page in session state if not exists
    if 'browse_page' not in st.session_state:
        st.session_state.browse_page = 1

    # Ensure page is within bounds
    if st.session_state.browse_page > total_pages:
        st.session_state.browse_page = total_pages

    # Pagination controls
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    with col1:
        if st.button("⏮️ First", disabled=st.session_state.browse_page == 1, use_container_width=True):
            st.session_state.browse_page = 1
            st.rerun()

    with col2:
        if st.button("◀️ Prev", disabled=st.session_state.browse_page == 1, use_container_width=True):
            st.session_state.browse_page -= 1
            st.rerun()

    with col3:
        st.markdown(f"<div style='text-align: center; padding: 8px;'>Page <strong>{st.session_state.browse_page}</strong> of <strong>{total_pages}</strong></div>", unsafe_allow_html=True)

    with col4:
        if st.button("Next ▶️", disabled=st.session_state.browse_page >= total_pages, use_container_width=True):
            st.session_state.browse_page += 1
            st.rerun()

    with col5:
        if st.button("Last ⏭️", disabled=st.session_state.browse_page >= total_pages, use_container_width=True):
            st.session_state.browse_page = total_pages
            st.rerun()

    page = st.session_state.browse_page
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_df))
    page_df = filtered_df.iloc[start_idx:end_idx]

    st.caption(f"Showing {start_idx + 1}-{end_idx} of {len(filtered_df)}")

    # Display grid
    cols_per_row = 5
    rows_needed = (len(page_df) + cols_per_row - 1) // cols_per_row

    for row_idx in range(rows_needed):
        cols = st.columns(cols_per_row)

        for col_idx in range(cols_per_row):
            plant_idx = row_idx * cols_per_row + col_idx

            if plant_idx >= len(page_df):
                break

            plant = page_df.iloc[plant_idx]
            plant_id = plant['plant_id']

            with cols[col_idx]:
                # Thumbnail
                thumbnail_path = data_loader.get_thumbnail_path(plant_id)
                if thumbnail_path.exists():
                    st.image(str(thumbnail_path), use_container_width=True)

                # Plant name (shortened)
                display_name = plant_id.replace('KW_T_423_', '')
                st.caption(f"**{display_name}**")

                # Dutch name
                if not pd.isna(plant.get('Huidige Nederlandse naam')):
                    st.caption(plant['Huidige Nederlandse naam'])

                # Taxonomy
                if not pd.isna(plant.get('genus')):
                    st.caption(f"*{plant['genus']}*")

                # Color palette
                colors = data_loader.get_color_palette(plant_id, ranking=color_ranking)
                if colors:
                    st.markdown(charts.create_color_palette_display(colors[:num_colors]), unsafe_allow_html=True)

                # View details
                if st.button("Details", key=f"detail_{plant_id}", use_container_width=True):
                    st.session_state.selected_plant_id = plant_id
                    st.switch_page("pages/4_Plant_Detail.py")
