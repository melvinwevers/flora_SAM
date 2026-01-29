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
        dutch_name = row.get('Huidige Nederlandse naam')
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
    The app provides **three different color rankings** for each plant:

    ### 1. Frequency Ranking
    Colors sorted by **area coverage** (most common first).
    - **Method**: K-means clustering extracts dominant colors, then sorts by pixel frequency
    - **Uses**: Understanding overall color composition
    - **Best for**: Finding plants with lots of green, identifying background colors

    ### 2. Perceptual Ranking
    Colors ranked by **perceptual distinctiveness** - a heuristic combining:
    - Frequency (40%): How much area the color covers
    - Saturation (30%): How vivid/pure the color is (from HSL color space)
    - Contrast (30%): Average **Delta E** distance to other colors in the palette (perceptually uniform LAB color space)
    - **Formula**: `0.4×frequency + 0.3×saturation + 0.3×contrast`
    - **Implementation**: Computed in `color_analysis.py::calculate_perceptual_weight()`
    - **Uses**: Finding colors that "pop" due to their inherent properties
    - **Best for**: Identifying vibrant or unusual colors regardless of location

    ### 3. Salience Ranking
    Colors ranked by **spatial attention** - measures visual salience:
    - **Method**: OpenCV Spectral Residual (`cv2.saliency.StaticSaliencySpectralResidual`) or edge-based fallback (Sobel operators + Gaussian blur)
    - **Process**:
      1. Compute saliency map once per image (values 0-1, highlighting attention-grabbing regions)
      2. For each color, create a mask of pixels within ±30 RGB tolerance
      3. Average saliency values at those pixel locations
    - **Implementation**: `color_analysis.py::compute_saliency_map()` and `calculate_saliency_weight()`
    - **Result**: Colors appearing in visually salient regions (edges, contrasts, focal points) rank higher
    - **Best for**: Identifying diagnostic botanical features the illustrator emphasized (flowers, fruits, distinctive structures)

    ### Color Representations
    All colors are provided in three color spaces:
    - **RGB**: Standard red-green-blue (0-255 per channel)
    - **HSL**: Hue (0-360°), Saturation (0-100%), Lightness (0-100%)
    - **LAB**: CIE L\\*a\\*b\\* perceptually uniform color space
      - L: Lightness (0-100)
      - a: Green (-) to Red (+) axis
      - b: Blue (-) to Yellow (+) axis

    ### Which Ranking to Use?

    - **Frequency**: "What colors make up most of this plant?"
    - **Perceptual**: "What colors are intrinsically striking?"
    - **Salience**: "What colors would I notice first when looking at this?"

    For botanical analysis, **salience** often works best as it captures what the
    illustrator emphasized - typically the diagnostic features like flowers or fruits.
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

