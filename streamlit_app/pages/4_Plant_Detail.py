"""
Plant Detail page - deep dive into a single plant
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils import data_loader, charts

st.set_page_config(page_title="Plant Detail", page_icon="🌱", layout="wide")

# Get plant_id from session state
plant_id = st.session_state.get('selected_plant_id', None)

if not plant_id:
    st.title("🌱 Plant Detail")
    st.info("Use the home page to search for a plant, or navigate here from another page.")
    st.stop()

# Load plant details
plant = data_loader.get_plant_details(plant_id)

if not plant:
    st.error(f"Plant {plant_id} not found")
    st.stop()

# Display plant with Dutch name
dutch_name = plant.get('Huidige Nederlandse naam')
if not pd.isna(dutch_name) and dutch_name:
    st.title(f"🌱 {dutch_name}")
    st.caption(f"Plant ID: {plant_id}")
else:
    st.title(f"🌱 {plant_id}")

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    # Large thumbnail
    thumbnail_path = data_loader.get_thumbnail_path(plant_id)

    if thumbnail_path.exists():
        st.image(str(thumbnail_path), use_container_width=True)
    else:
        st.warning("Thumbnail not available")

with col2:
    # Metadata
    st.subheader("Taxonomy")

    if not pd.isna(plant.get('family')):
        st.write(f"**Family:** {plant['family']}")
    else:
        st.write("**Family:** Unknown")

    if not pd.isna(plant.get('genus')):
        st.write(f"**Genus:** {plant['genus']}")
    else:
        st.write("**Genus:** Unknown")

    if not pd.isna(plant.get('species')):
        st.write(f"**Species:** {plant['species']}")

    st.subheader("Image Properties")

    st.write(f"**Dimensions:** {plant.get('width', '?')} × {plant.get('height', '?')} px")
    st.write(f"**Area:** {plant.get('area_pixels', 0):,} pixels")

# Color palettes
st.header("Color Palettes")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Frequency Ranking")
    st.caption("Colors ranked by area coverage")

    colors_freq = data_loader.get_color_palette(plant_id, ranking='frequency')
    if colors_freq:
        st.markdown(charts.create_color_palette_display(colors_freq), unsafe_allow_html=True)

with col2:
    st.subheader("Visual Importance Ranking")
    st.caption("Colors ranked by visual saliency")

    colors_visual = data_loader.get_color_palette(plant_id, ranking='visual')
    if colors_visual:
        st.markdown(charts.create_color_palette_display(colors_visual), unsafe_allow_html=True)

# Cluster membership
st.header("Visual Cluster Membership")

if 'dinov2' in plant and not pd.isna(plant['dinov2']):
    cluster_id = int(plant['dinov2'])
    if cluster_id >= 0:
        st.metric("DINOv2 Cluster", f"Cluster {cluster_id}")
    else:
        st.info("This plant was classified as noise (doesn't fit into any cluster)")
else:
    st.info("No cluster assignment available")

# Similar plants
st.header("Visually Similar Plants")

st.caption("Plants in the same DINOv2 cluster")

similar = data_loader.get_similar_plants(plant_id, model='dinov2', limit=6)

if len(similar) > 0:
    cols = st.columns(min(6, len(similar)))

    for idx, (col, row) in enumerate(zip(cols, similar.itertuples())):
        with col:
            similar_id = row.plant_id
            similar_thumb = data_loader.get_thumbnail_path(similar_id)

            if similar_thumb.exists():
                st.image(str(similar_thumb), use_container_width=True)

            st.caption(f"**{similar_id}**")

            if not pd.isna(row.family):
                st.caption(f"{row.family}")

            if st.button("View", key=f"similar_{similar_id}"):
                st.session_state.selected_plant_id = similar_id
                st.rerun()
else:
    st.info("No similar plants found (plant may be in noise cluster)")

# Related plants
st.header("Taxonomically Related Plants")

st.caption("Plants in the same genus")

related = data_loader.get_related_plants(plant_id, limit=6)

if len(related) > 0:
    cols = st.columns(min(6, len(related)))

    for idx, (col, row) in enumerate(zip(cols, related.itertuples())):
        with col:
            related_id = row.plant_id
            related_thumb = data_loader.get_thumbnail_path(related_id)

            if related_thumb.exists():
                st.image(str(related_thumb), use_container_width=True)

            st.caption(f"**{related_id}**")

            if not pd.isna(row.species):
                st.caption(f"{row.species}")

            if st.button("View", key=f"related_{related_id}"):
                st.session_state.selected_plant_id = related_id
                st.rerun()
else:
    st.info("No related plants found (genus unknown or unique)")

# Navigation back
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("← Browse by Taxonomy"):
        st.switch_page("pages/1_Browse_by_Taxonomy.py")

with col2:
    if st.button("← Visual Clusters"):
        st.switch_page("pages/2_Visual_Clusters.py")

with col3:
    if st.button("← Color Analysis"):
        st.switch_page("pages/3_Color_Analysis.py")
