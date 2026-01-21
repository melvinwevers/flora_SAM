"""
Visual Clusters page - explore AI-discovered similarity groups
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils import data_loader, charts

st.set_page_config(page_title="Visual Clusters", page_icon="🔍", layout="wide")

st.title("🔍 Visual Clusters")

st.markdown("""
Explore plants grouped by visual similarity using deep learning embeddings.
Different models capture different aspects of plant appearance.
""")

# Sidebar controls
st.sidebar.header("Settings")

# Model selection
model_options = {
    'DINOv2 (Vision Transformer)': 'dinov2',
    'CLIP (Vision-Language)': 'clip',
    'PlantNet (Botanical)': 'plantnet',
    'Combined (All models)': 'combined'
}

selected_model_name = st.sidebar.selectbox("Embedding Model", list(model_options.keys()))
selected_model = model_options[selected_model_name]

# Get cluster statistics
cluster_stats = data_loader.get_cluster_stats(selected_model)

if not cluster_stats:
    st.error(f"No cluster data available for {selected_model_name}")
    st.stop()

# Cluster selection
cluster_ids = sorted(cluster_stats.keys())
cluster_options = {f"Cluster {cid} ({cluster_stats[cid]} plants)": cid for cid in cluster_ids}

selected_cluster_name = st.sidebar.selectbox("Cluster", list(cluster_options.keys()))
selected_cluster_id = cluster_options[selected_cluster_name]

# Display model info
st.header(f"Model: {selected_model_name}")

model_descriptions = {
    'dinov2': "Self-supervised Vision Transformer trained on images. Captures general visual patterns.",
    'clip': "Vision-Language model from OpenAI. Understands semantic concepts and image-text relationships.",
    'plantnet': "Botanical specialist pretrained on plant species. Leverages domain-specific knowledge.",
    'combined': "Concatenation of all three embeddings. Best overall clustering performance."
}

st.info(model_descriptions.get(selected_model, ""))

# Display cluster summary
st.header(f"Cluster {selected_cluster_id} Summary")

filtered_df = data_loader.get_plants_by_cluster(selected_model, selected_cluster_id)

col1, col2, col3 = st.columns(3)

col1.metric("Plants in Cluster", len(filtered_df))

# Taxonomic composition
if 'family' in filtered_df.columns:
    family_counts = filtered_df['family'].value_counts()
    if len(family_counts) > 0:
        dominant_family = family_counts.index[0]
        dominant_count = family_counts.iloc[0]
        col2.metric("Dominant Family", dominant_family)
        col3.metric("Taxonomic Purity", f"{(dominant_count / len(filtered_df) * 100):.1f}%")

# Taxonomic breakdown
if 'family' in filtered_df.columns:
    st.subheader("Taxonomic Composition")

    family_counts = filtered_df['family'].value_counts().head(10)

    if len(family_counts) > 0:
        fig = charts.create_bar_chart(
            family_counts,
            "Family",
            "Count",
            f"Top Families in Cluster {selected_cluster_id}"
        )
        st.plotly_chart(fig, use_container_width=True)

# Display plants grid
st.subheader("Plants in this Cluster")

# Color ranking toggle
color_ranking = st.radio(
    "Color Ranking",
    options=["frequency", "visual"],
    format_func=lambda x: "Frequency" if x == "frequency" else "Visual Importance",
    horizontal=True
)

# Pagination
plants_per_page = 20
total_pages = (len(filtered_df) + plants_per_page - 1) // plants_per_page

page = st.selectbox(
    "Page",
    range(1, total_pages + 1),
    format_func=lambda x: f"Page {x} of {total_pages}"
)

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
                st.warning("Thumbnail not found")

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

# Cluster statistics
st.subheader("All Clusters Overview")

st.markdown(f"**Total clusters in {selected_model_name}:** {len(cluster_stats)}")

# Show cluster sizes
cluster_size_df = pd.DataFrame(list(cluster_stats.items()), columns=['Cluster', 'Size'])
cluster_size_df = cluster_size_df.sort_values('Size', ascending=False)

fig = charts.create_bar_chart(
    dict(zip(cluster_size_df['Cluster'], cluster_size_df['Size'])),
    "Cluster",
    "Number of Plants",
    f"Cluster Sizes ({selected_model_name})"
)
st.plotly_chart(fig, use_container_width=True)
