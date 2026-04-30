"""
Flora Batava Plant Explorer

A Streamlit app for exploring botanical illustrations from the Flora Batava collection
through taxonomy, visual clusters, and color analysis.
"""
import streamlit as st
import pandas as pd
import random
from utils import data_loader

# Page configuration
st.set_page_config(
    page_title="Flora Batava Explorer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Flora Batava Plant Explorer - Browse botanical illustrations by taxonomy, visual similarity, and color analysis."
    }
)

# Home page
st.title("🌿 Flora Batava Plant Explorer")

# Hero search section
search_col, random_col = st.columns([3, 1])

with search_col:
    # Load plants for search
    plants_df = data_loader.load_plants_metadata()

    # Create autocomplete options with mapping
    autocomplete_options = [""]  # Empty option first
    option_to_plant_id = {}  # Map display name to plant_id

    for _, row in plants_df.iterrows():
        plant_id = row['plant_id']
        dutch_name = row.get('dutch_name')
        genus = row.get('genus')
        species = row.get('species')

        # Build Latin name
        latin_parts = []
        if pd.notna(genus):
            latin_parts.append(genus)
        if pd.notna(species):
            latin_parts.append(species)
        latin_name = " ".join(latin_parts) if latin_parts else None

        # Build display name
        if pd.notna(dutch_name) and dutch_name:
            if latin_name:
                display_name = f"{dutch_name} ({latin_name})"
            else:
                display_name = dutch_name
        elif latin_name:
            display_name = latin_name
        else:
            display_name = f"Plant {plant_id.replace('KW_T_423_', '')}"

        autocomplete_options.append(display_name)
        option_to_plant_id[display_name] = plant_id

    # Autocomplete search using selectbox
    selected_plant = st.selectbox(
        "Search plants",
        options=autocomplete_options,
        placeholder="Type to search by name...",
        label_visibility="collapsed",
        key="plant_search"
    )

    # Auto-navigate when plant is selected
    if selected_plant and selected_plant != "":
        selected_plant_id = option_to_plant_id[selected_plant]
        st.session_state.selected_plant_id = selected_plant_id
        st.switch_page("pages/Plant_Detail.py")

with random_col:
    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)  # Align with search input
    if st.button("🎲 Random Plant", use_container_width=True):
        random_plant_id = random.choice(plants_df['plant_id'].tolist())
        st.session_state.selected_plant_id = random_plant_id
        st.switch_page("pages/Plant_Detail.py")

st.markdown("---")

st.markdown("""
Welcome to the Flora Batava Plant Explorer! This interactive application allows you to explore
historical botanical illustrations through multiple lenses:

- **Browse by Taxonomy**: Navigate through plant families, genera, and species
- **Visual Clusters**: Explore clusters based on visual similarity groups
- **Color Analysis**: Analyze color patterns and their relationships to taxonomy and clusters
- **Plant Details**: Deep dive into individual plants with colors, clusters, and related specimens

Use the sidebar to navigate between different views, or use the search above to find a specific plant.
""")

with st.expander("How are colors ranked?"):
    st.markdown("""
    The app provides **two color rankings** for each plant:

    ### 1. Frequency
    Colors sorted by **area coverage** — the most common color first.

    **What it measures:** Percentage of the illustration occupied by each color.

    **Best for:** Understanding the overall color composition and what the illustrator
    painted most prominently.

    ### 2. Vividness (Chroma)
    Colors sorted by **chromatic intensity** — the most vivid colors first.

    **What it measures:** LAB chroma = √(a² + b²), a perceptually uniform measure of
    color vividness. Higher chroma = more saturated, vibrant color.

    **Why this works for botanical illustrations:** Historical botanical illustrators
    intentionally used vivid watercolor pigments to emphasize diagnostic features like
    flowers, fruits, and berries. By ranking colors by vividness, we reveal what the
    18th-century artist chose to highlight.

    **Best for:** Finding flowers, fruits, and other colorful features that may occupy
    small areas (< 1% of pixels) but were painted with vivid pigments.

    **Research basis:** Chroma ranking is based on vision research showing that color
    saturation is a key factor in visual attention ([Pal et al. 2012](https://dl.acm.org/doi/10.1145/2425333.2425357)),
    and historical practice where botanical illustrators intentionally used vivid watercolors
    to emphasize diagnostic features ([Bauer's pigment studies](https://www.researchgate.net/publication/230083140)).

    ### Which ranking to use?

    | Ranking | Question it answers | Example |
    |---|---|---|
    | **Frequency** | "What colors dominate?" | Green leaves (70%), brown stems (15%), yellow flower (0.5%) |
    | **Vividness** | "What did the artist paint most vividly?" | Yellow flower (chroma 50), green leaves (chroma 18), brown stems (chroma 12) |

    **Tip:** For finding characteristic flower or fruit colors, use **Vividness**.
    For understanding the overall illustration composition, use **Frequency**.
    """)

# Load cluster data
clusters_df = data_loader.load_cluster_assignments()

st.markdown("---")

# Load and display summary statistics
st.header("Dataset Overview")

col1, col2, col3, col4, col5 = st.columns(5)

# Statistics
total_plants = len(plants_df)
plants_with_taxonomy = plants_df['family'].notna().sum()
n_families = plants_df['family'].nunique()
n_genera = plants_df['genus'].nunique()

col1.metric("Total Plants", f"{total_plants:,}")
col2.metric("With Taxonomy", f"{plants_with_taxonomy:,}")
col3.metric("Families", f"{n_families}")
col4.metric("Genera", f"{n_genera}")

if 'dinov2' in clusters_df.columns:
    n_clusters = clusters_df[clusters_df['dinov2'] >= 0]['dinov2'].nunique()
    col5.metric("DINOv2 Clusters", f"{n_clusters}")


st.markdown("""

- Source: Flora Batava historical botanical illustration collection
- Segmentation: Meta's Segment Anything Model (SAM)
- Color Analysis: K-means clustering in perceptually uniform LAB color space
- Embeddings: DINOv2 Vision Transformer for visual similarity clustering
""")

# Sidebar info

