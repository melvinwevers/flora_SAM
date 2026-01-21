"""
Browse by Taxonomy page - navigate through family/genus hierarchy
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils import data_loader, charts

st.set_page_config(page_title="Browse by Taxonomy", page_icon="🌿", layout="wide")

st.title("🌿 Browse by Taxonomy")

st.markdown("""
Navigate through the botanical classification of plants in the Flora Batava collection.
Filter by family and genus to explore specific taxonomic groups.
""")

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

# Species search (optional)
species_search = st.sidebar.text_input("Species (optional)", "")

# Get filtered plants
family_filter = None if selected_family == "All Families" else selected_family
genus_filter = None if selected_genus == "All Genera" else selected_genus
species_filter = species_search if species_search else None

filtered_df = data_loader.get_plants_by_taxonomy(
    family=family_filter,
    genus=genus_filter,
    species=species_filter
)

# Display summary
st.header("Selection Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Plants Found", len(filtered_df))

if family_filter:
    col2.metric("Family", family_filter)
if genus_filter:
    col3.metric("Genus", genus_filter)

# Color ranking toggle
color_ranking = st.radio(
    "Color Ranking Method",
    options=["frequency", "visual"],
    format_func=lambda x: "Frequency (by area)" if x == "frequency" else "Visual Importance",
    horizontal=True
)

# Show cluster distribution if we have cluster data
if len(filtered_df) > 0 and 'dinov2' in filtered_df.columns:
    st.header("Visual Cluster Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig = charts.create_cluster_distribution_pie(
            filtered_df,
            'dinov2',
            f"DINOv2 Clusters ({selected_family if family_filter else 'All'})"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = charts.create_cluster_distribution_pie(
            filtered_df,
            'combined',
            f"Combined Clusters ({selected_family if family_filter else 'All'})"
        )
        st.plotly_chart(fig, use_container_width=True)

# Display plants grid
st.header("Plants")

if len(filtered_df) == 0:
    st.warning("No plants found with the selected filters.")
else:
    # Pagination
    plants_per_page = 20
    total_pages = (len(filtered_df) + plants_per_page - 1) // plants_per_page

    page = st.selectbox("Page", range(1, total_pages + 1), format_func=lambda x: f"Page {x} of {total_pages}")

    start_idx = (page - 1) * plants_per_page
    end_idx = start_idx + plants_per_page
    page_df = filtered_df.iloc[start_idx:end_idx]

    # Display plants in grid
    cols_per_row = 4
    rows = (len(page_df) + cols_per_row - 1) // cols_per_row

    for row_idx in range(rows):
        cols = st.columns(cols_per_row)

        for col_idx in range(cols_per_row):
            plant_idx = row_idx * cols_per_row + col_idx

            if plant_idx >= len(page_df):
                break

            plant = page_df.iloc[plant_idx]
            plant_id = plant['plant_id']
            thumbnail_path = data_loader.get_thumbnail_path(plant_id)

            with cols[col_idx]:
                # Display thumbnail
                if thumbnail_path.exists():
                    st.image(str(thumbnail_path), use_container_width=True)
                else:
                    st.warning(f"Thumbnail not found")

                # Plant info
                st.caption(f"**{plant_id}**")

                if not pd.isna(plant.get('family')):
                    st.caption(f"Family: {plant['family']}")
                if not pd.isna(plant.get('genus')):
                    st.caption(f"Genus: {plant['genus']}")

                # Color palette
                colors = data_loader.get_color_palette(plant_id, ranking=color_ranking)
                if colors:
                    st.markdown(charts.create_color_palette_display(colors[:3]), unsafe_allow_html=True)

                # Link to detail page
                if st.button("View Details", key=f"detail_{plant_id}"):
                    st.session_state.selected_plant_id = plant_id
                    st.switch_page("pages/4_Plant_Detail.py")
