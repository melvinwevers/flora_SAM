"""
Flora Batava Plant Explorer

A Streamlit app for exploring botanical illustrations from the Flora Batava collection
through taxonomy, visual clusters, and color analysis.
"""
import streamlit as st
import random
import pandas as pd
from utils import data_loader

# Page configuration
st.set_page_config(
    page_title="Flora Batava Explorer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page
st.title("🌿 Flora Batava Plant Explorer")

st.markdown("""
Welcome to the Flora Batava Plant Explorer! This interactive application allows you to explore
historical botanical illustrations through multiple lenses:

- **Browse by Taxonomy**: Navigate through plant families, genera, and species
- **Visual Clusters**: Explore AI-discovered similarity groups using DINOv2
- **Color Analysis**: Analyze color patterns and their relationships to taxonomy and clusters
- **Plant Details**: Deep dive into individual plants with colors, clusters, and related specimens

Use the sidebar to navigate between different views.
""")

with st.expander("ℹ️ How is visual salience calculated?"):
    st.markdown("""
    The app shows two color rankings for each plant:

    1. **Frequency Ranking**: Colors sorted by area coverage (most common first)
    2. **Visual Importance Ranking**: Colors sorted by visual salience

    **Visual Salience Score** combines three factors to identify colors that catch the eye:

    - **Frequency (40%)**: How much area the color covers
    - **Saturation (30%)**: How vivid/pure the color is (more saturated = more noticeable)
    - **Contrast (30%)**: How different the color is from other colors in the image

    **Formula**: `importance = 0.4×frequency + 0.3×saturation + 0.3×contrast`

    This means a small but highly saturated, contrasting color (like a bright red flower center)
    can rank higher than a large but dull color (like grayish-green leaves).

    **Why this matters**: Visual salience better captures which colors draw human attention,
    making it useful for finding plants with distinctive color features like bright flowers,
    vibrant berries, or striking foliage patterns.
    """)

# Load data
plants_df = data_loader.load_plants_metadata()
clusters_df = data_loader.load_cluster_assignments()

# Hero search section
st.markdown("---")
st.markdown("### Start Exploring")

search_col, button_col = st.columns([3, 1])

with search_col:
    search_query = st.text_input(
        "Search for a plant",
        "",
        placeholder="Search by plant ID or name...",
        label_visibility="collapsed"
    )

with button_col:
    random_button = st.button("🎲 Random Plant", type="primary", use_container_width=True)

# Handle search query
if search_query:
    query_lower = search_query.lower()
    matches = plants_df[
        plants_df['plant_id'].str.lower().str.contains(query_lower) |
        plants_df['Huidige Nederlandse naam'].fillna('').str.lower().str.contains(query_lower)
    ]

    if len(matches) > 0:
        # Limit to 15 matches
        display_matches = matches.head(15)

        # Create display options
        display_options = []
        for _, plant in display_matches.iterrows():
            plant_id = plant['plant_id']
            dutch_name = plant.get('Huidige Nederlandse naam', '')
            if pd.notna(dutch_name) and dutch_name:
                display_options.append(f"{plant_id} - {dutch_name}")
            else:
                display_options.append(plant_id)

        # Show match count if limited
        label = f"Found {len(matches)} plants"
        if len(matches) > 15:
            label = f"Showing 15 of {len(matches)} matches - keep typing to narrow"

        selected_display = st.selectbox(
            label,
            display_options,
            key="home_search_select"
        )

        if st.button("Go to plant", type="primary"):
            # Extract plant_id from display string
            selected_id = selected_display.split(' - ')[0]
            st.session_state.selected_plant_id = selected_id
            st.switch_page("pages/4_Plant_Detail.py")
    else:
        st.warning("No plants found")

# Handle random plant button
if random_button:
    plant_options = plants_df['plant_id'].tolist()
    st.session_state.selected_plant_id = random.choice(plant_options)
    st.switch_page("pages/4_Plant_Detail.py")

st.markdown("---")

# Load and display summary statistics
st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

# Statistics
total_plants = len(plants_df)
plants_with_taxonomy = plants_df['family'].notna().sum()
n_families = plants_df['family'].nunique()
n_genera = plants_df['genus'].nunique()

col1.metric("Total Plants", f"{total_plants:,}")
col2.metric("With Taxonomy", f"{plants_with_taxonomy:,}")
col3.metric("Families", f"{n_families}")
col4.metric("Genera", f"{n_genera}")

# Clustering info
st.header("Visual Clustering")

if 'dinov2' in clusters_df.columns:
    n_clusters = clusters_df[clusters_df['dinov2'] >= 0]['dinov2'].nunique()
    st.metric("DINOv2 Clusters", f"{n_clusters} clusters")

st.markdown("""
---
### Getting Started

Use the **sidebar** to navigate to different pages:

1. **Browse by Taxonomy** - Start here to explore plants by their botanical classification
2. **Visual Clusters** - See how AI models group plants by visual similarity
3. **Color Analysis** - Compare color patterns across taxonomy and clusters
4. **Plant Detail** - (Navigate from other pages to see individual plant details)

---

**About the Data:**
- Source: Flora Batava historical botanical illustration collection
- Segmentation: Meta's Segment Anything Model (SAM)
- Color Analysis: K-means clustering in perceptually uniform LAB color space
- Embeddings: DINOv2 Vision Transformer for visual similarity clustering
""")

# Sidebar info
st.sidebar.title("Navigation")
st.sidebar.info("""
Select a page from the sidebar menu to begin exploring the Flora Batava collection.

**Quick Tips:**
- Start with Browse by Taxonomy to filter by family/genus
- Use Visual Clusters to discover unexpected similarities
- Check Color Analysis to see how colors relate to groups
""")
