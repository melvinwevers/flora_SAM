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
Plants grouped by visual similarity using DINOv2 deep learning. Browse through clusters to discover plants that look alike.
""")

# Get cluster statistics
selected_model = 'dinov2'
cluster_stats = data_loader.get_cluster_stats(selected_model)

if not cluster_stats:
    st.error(f"No cluster data available for DINOv2")
    st.stop()

# All Clusters Overview
st.header("All Clusters")

cluster_ids = sorted(cluster_stats.keys())
cluster_cols = st.columns(min(len(cluster_ids), 8))

for idx, cluster_id in enumerate(cluster_ids):
    col_idx = idx % len(cluster_cols)
    with cluster_cols[col_idx]:
        if st.button(
            f"**{cluster_id}**\n{cluster_stats[cluster_id]} plants",
            key=f"cluster_{cluster_id}",
            use_container_width=True
        ):
            st.session_state.selected_cluster_id = cluster_id
            st.rerun()

st.markdown("---")

# Cluster selection
if 'selected_cluster_id' not in st.session_state:
    st.session_state.selected_cluster_id = cluster_ids[0]

selected_cluster_id = st.session_state.selected_cluster_id

# Display cluster
st.header(f"Cluster {selected_cluster_id}")

filtered_df = data_loader.get_plants_by_cluster(selected_model, selected_cluster_id)

st.caption(f"{len(filtered_df)} plants in this cluster")

# Display options
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    color_ranking = st.radio(
        "Color Ranking",
        options=["frequency", "visual"],
        format_func=lambda x: "By Area" if x == "frequency" else "Visual Importance",
        horizontal=True
    )

with col2:
    num_colors = st.slider("Colors shown", 3, 5, 5)

with col3:
    plants_per_page = st.selectbox("Per page", [20, 40, 60], index=1)

# Pagination
total_pages = max(1, (len(filtered_df) + plants_per_page - 1) // plants_per_page)

# Initialize page in session state
if 'cluster_page' not in st.session_state:
    st.session_state.cluster_page = 1

# Reset page if cluster changed
if st.session_state.cluster_page > total_pages:
    st.session_state.cluster_page = 1

# Pagination controls
col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

with col1:
    if st.button("⏮️ First", disabled=st.session_state.cluster_page == 1, use_container_width=True):
        st.session_state.cluster_page = 1
        st.rerun()

with col2:
    if st.button("◀️ Prev", disabled=st.session_state.cluster_page == 1, use_container_width=True):
        st.session_state.cluster_page -= 1
        st.rerun()

with col3:
    st.markdown(f"<div style='text-align: center; padding: 8px;'>Page <strong>{st.session_state.cluster_page}</strong> of <strong>{total_pages}</strong></div>", unsafe_allow_html=True)

with col4:
    if st.button("Next ▶️", disabled=st.session_state.cluster_page >= total_pages, use_container_width=True):
        st.session_state.cluster_page += 1
        st.rerun()

with col5:
    if st.button("Last ⏭️", disabled=st.session_state.cluster_page >= total_pages, use_container_width=True):
        st.session_state.cluster_page = total_pages
        st.rerun()

page = st.session_state.cluster_page

start_idx = (page - 1) * plants_per_page
end_idx = min(start_idx + plants_per_page, len(filtered_df))
page_df = filtered_df.iloc[start_idx:end_idx]

st.caption(f"Showing {start_idx + 1}-{end_idx} of {len(filtered_df)}")

# Display plants in grid
cols_per_row = 5
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

            # Plant name (shortened)
            display_name = plant_id.replace('KW_T_423_', '')
            st.caption(f"**{display_name}**")

            # Dutch name
            if not pd.isna(plant.get('Huidige Nederlandse naam')):
                st.caption(plant['Huidige Nederlandse naam'])

            # Genus
            if not pd.isna(plant.get('genus')):
                st.caption(f"*{plant['genus']}*")

            # Color palette
            colors = data_loader.get_color_palette(plant_id, ranking=color_ranking)
            if colors:
                st.markdown(charts.create_color_palette_display(colors[:num_colors]), unsafe_allow_html=True)

            # Link to detail page
            if st.button("Details", key=f"detail_{plant_id}", use_container_width=True):
                st.session_state.selected_plant_id = plant_id
                st.switch_page("pages/4_Plant_Detail.py")
