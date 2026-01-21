"""
Flora Batava Plant Explorer

A Streamlit app for exploring botanical illustrations from the Flora Batava collection
through taxonomy, visual clusters, and color analysis.
"""
import streamlit as st
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
2,241 historical botanical illustrations through multiple lenses:

- **Browse by Taxonomy**: Navigate through plant families, genera, and species
- **Visual Clusters**: Explore AI-discovered similarity groups using DINOv2, CLIP, and PlantNet
- **Color Analysis**: Analyze color patterns and their relationships to taxonomy and clusters
- **Plant Details**: Deep dive into individual plants with colors, clusters, and related specimens

Use the sidebar to navigate between different views.
""")

# Load and display summary statistics
st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

# Load data
plants_df = data_loader.load_plants_metadata()
clusters_df = data_loader.load_cluster_assignments()

# Statistics
total_plants = len(plants_df)
plants_with_taxonomy = plants_df['family'].notna().sum()
n_families = plants_df['family'].nunique()
n_genera = plants_df['genus'].nunique()

col1.metric("Total Plants", f"{total_plants:,}")
col2.metric("With Taxonomy", f"{plants_with_taxonomy:,}")
col3.metric("Families", f"{n_families}")
col4.metric("Genera", f"{n_genera}")

# Clustering models info
st.header("Clustering Models")

col1, col2, col3, col4 = st.columns(4)

models = {
    'DINOv2': 'dinov2',
    'CLIP': 'clip',
    'PlantNet': 'plantnet',
    'Combined': 'combined'
}

for col, (name, model_col) in zip([col1, col2, col3, col4], models.items()):
    if model_col in clusters_df.columns:
        n_clusters = clusters_df[clusters_df[model_col] >= 0][model_col].nunique()
        col.metric(name, f"{n_clusters} clusters")

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
- Embeddings: DINOv2 (Vision Transformer), CLIP (Vision-Language), PlantNet (Botanical)
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
