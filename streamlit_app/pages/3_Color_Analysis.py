"""
Browse by Color - explore plants grouped by dominant colors
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils import data_loader, charts
from utils.color_utils import COLOR_BUCKETS, get_color_bucket_info

st.set_page_config(page_title="Browse by Color", page_icon="🎨", layout="wide")

st.title("🎨 Browse by Color")

st.markdown("""
Pick any color and find plants with similar dominant colors.
""")

# Color picker section
st.header("Pick a Color")

col1, col2 = st.columns([1, 4])

with col1:
    # Initialize selected color in session state
    if 'custom_color' not in st.session_state:
        st.session_state.custom_color = '#4CAF50'  # Default green

    # Color picker
    selected_color = st.color_picker("Choose color", st.session_state.custom_color, key="color_picker_main")

    if selected_color != st.session_state.custom_color:
        st.session_state.custom_color = selected_color
        st.session_state.color_page = 1  # Reset pagination
        st.rerun()



with col2:
    st.header("Settings")

    col2a, col2b = st.columns(2)

    with col2a:
        # Color ranking selection
        color_ranking = st.radio(
            "Color Ranking Method",
            options=["frequency", "perceptual", "saliency"],
            format_func=lambda x: {"frequency": "Frequency", "perceptual": "Perceptual", "saliency": "Salience"}[x],
            help="Frequency: area coverage | Perceptual: color properties | Salience: spatial attention"
        )

    with col2b:
        # Similarity threshold slider (inverted so higher = more similar)
        similarity = st.slider(
            "Color similarity",
            min_value=0,
            max_value=100,
            value=10,
            help="Higher = include more plants with similar colors. Lower = only very close matches."
        )
        # Convert to distance (invert the scale)
        max_distance = 100 - similarity

st.markdown("---")

# Get plants by color similarity
filtered_df = data_loader.get_plants_by_color_similarity(
    selected_color,
    ranking=color_ranking,
    max_distance=max_distance
)

st.subheader(f"Found {len(filtered_df)} plants with similar colors")

# Display options
col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

with col1:
    show_palette = st.checkbox("Show color palettes", value=True)

num_colors = 5

with col3:
    palette_style = "squares"

with col4:
    plants_per_page = st.selectbox("Per page", [20, 40, 60], index=1)

# Pagination
total_pages = max(1, (len(filtered_df) + plants_per_page - 1) // plants_per_page)

# Initialize page in session state
if 'color_page' not in st.session_state:
    st.session_state.color_page = 1

# Reset page if color bucket changed and page is out of bounds
if st.session_state.color_page > total_pages:
    st.session_state.color_page = 1

# Pagination controls
col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

with col1:
    if st.button("⏮️ First", disabled=st.session_state.color_page == 1, use_container_width=True):
        st.session_state.color_page = 1
        st.rerun()

with col2:
    if st.button("◀️ Prev", disabled=st.session_state.color_page == 1, use_container_width=True):
        st.session_state.color_page -= 1
        st.rerun()

with col3:
    st.markdown(f"<div style='text-align: center; padding: 8px;'>Page <strong>{st.session_state.color_page}</strong> of <strong>{total_pages}</strong></div>", unsafe_allow_html=True)

with col4:
    if st.button("Next ▶️", disabled=st.session_state.color_page >= total_pages, use_container_width=True):
        st.session_state.color_page += 1
        st.rerun()

with col5:
    if st.button("Last ⏭️", disabled=st.session_state.color_page >= total_pages, use_container_width=True):
        st.session_state.color_page = total_pages
        st.rerun()

page = st.session_state.color_page
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
            if show_palette:
                colors = data_loader.get_color_palette(plant_id, ranking=color_ranking)
                if colors:
                    st.markdown(charts.create_color_palette_display(colors[:num_colors]), unsafe_allow_html=True)
            st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
            # Link to detail page
            if st.button("Details", key=f"detail_{plant_id}", use_container_width=True):
                st.session_state.selected_plant_id = plant_id
                st.switch_page("pages/Plant_Detail.py")
