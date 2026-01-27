"""
Data loading utilities with Streamlit caching.
"""
import streamlit as st
import pandas as pd
import json
from pathlib import Path

# Paths relative to streamlit_app directory
DATA_DIR = Path(__file__).parent.parent / "data"
THUMBNAILS_DIR = Path(__file__).parent.parent / "thumbnails"

@st.cache_data
def load_dutch_names():
    """Load Flora Batava Index with Dutch names.

    The Flora Batava Index rows align with plants_metadata rows by index.
    """
    df = pd.read_excel(DATA_DIR / "Flora Batava Index.xlsx")

    # Select relevant columns - the index alignment is maintained
    result = df[['Huidige Nederlandse naam', 'Oude Nederlandse naam in Flora Batava',
                 'Huidige botanische naam', 'Link naar scans in bladerboek KB']].copy()

    return result

@st.cache_data
def load_plants_metadata():
    """Load plants metadata CSV with colors and taxonomy."""
    df = pd.read_csv(DATA_DIR / "plants_metadata.csv")

    # Merge with Dutch names by index (rows align between files)
    dutch_names = load_dutch_names()

    # Reset index to ensure alignment, then concat along columns
    df_reset = df.reset_index(drop=True)
    dutch_reset = dutch_names.reset_index(drop=True)

    # Concatenate side by side (since rows align)
    df_merged = pd.concat([df_reset, dutch_reset], axis=1)

    return df_merged

@st.cache_data
def load_cluster_assignments():
    """Load cluster assignments for all models."""
    df = pd.read_csv(DATA_DIR / "cluster_assignments.csv")
    return df

@st.cache_data
def load_comparison_metrics():
    """Load precomputed comparison metrics."""
    with open(DATA_DIR / "comparison_metrics.json") as f:
        return json.load(f)

@st.cache_data
def load_merged_data():
    """Load and merge all data sources."""
    plants = load_plants_metadata()
    clusters = load_cluster_assignments()

    # Merge on plant_id
    merged = plants.merge(clusters, on='plant_id', how='left')
    return merged

def get_thumbnail_path(plant_id):
    """Get thumbnail path for a plant_id."""
    # Thumbnails are named {plant_id}.tif.jpg
    return THUMBNAILS_DIR / f"{plant_id}.tif.jpg"

def get_families():
    """Get list of unique families with counts."""
    df = load_plants_metadata()
    families = df[df['family'].notna()]['family'].value_counts()
    return families

def get_genera(family=None):
    """Get list of unique genera, optionally filtered by family."""
    df = load_plants_metadata()

    if family:
        df = df[df['family'] == family]

    genera = df[df['genus'].notna()]['genus'].value_counts()
    return genera

def get_plants_by_taxonomy(family=None, genus=None, species=None):
    """Get plants filtered by taxonomy."""
    df = load_merged_data()

    if family:
        df = df[df['family'] == family]
    if genus:
        df = df[df['genus'] == genus]
    if species:
        df = df[df['species'] == species]

    return df

def get_plants_by_cluster(model, cluster_id):
    """Get plants in a specific cluster."""
    df = load_merged_data()

    if model not in df.columns:
        return pd.DataFrame()

    return df[df[model] == cluster_id]

def get_cluster_stats(model):
    """Get cluster statistics for a model."""
    df = load_cluster_assignments()

    if model not in df.columns:
        return {}

    # Count plants per cluster, excluding noise (-1)
    counts = df[df[model] >= 0][model].value_counts().to_dict()
    return counts

def get_plant_details(plant_id):
    """Get detailed information for a single plant."""
    df = load_merged_data()
    plant = df[df['plant_id'] == plant_id]

    if len(plant) == 0:
        return None

    return plant.iloc[0].to_dict()

def get_similar_plants(plant_id, model='dinov2', limit=6):
    """Get plants similar to the given plant (same cluster)."""
    plant = get_plant_details(plant_id)

    if not plant or model not in plant:
        return pd.DataFrame()

    cluster_id = plant[model]

    if cluster_id < 0:  # Noise
        return pd.DataFrame()

    # Get all plants in same cluster, excluding the query plant
    similar = get_plants_by_cluster(model, cluster_id)
    similar = similar[similar['plant_id'] != plant_id]

    return similar.head(limit)

def get_related_plants(plant_id, limit=6):
    """Get taxonomically related plants (same genus)."""
    plant = get_plant_details(plant_id)

    if not plant or pd.isna(plant.get('genus')):
        return pd.DataFrame()

    genus = plant['genus']

    # Get all plants in same genus, excluding the query plant
    df = load_merged_data()
    related = df[(df['genus'] == genus) & (df['plant_id'] != plant_id)]

    return related.head(limit)

def get_color_palette(plant_id, ranking='frequency'):
    """
    Get color palette for a plant.

    Args:
        plant_id: Plant identifier
        ranking: One of 'frequency', 'perceptual', or 'saliency'
    """
    plant = get_plant_details(plant_id)

    if not plant:
        return []

    # Map ranking names to column prefixes
    prefix_map = {
        'frequency': 'color_freq',
        'perceptual': 'color_perceptual',
        'saliency': 'color_saliency',
        'visual': 'color_saliency'  # backwards compatibility
    }

    prefix = prefix_map.get(ranking, 'color_freq')

    colors = []
    for i in range(1, 6):
        hex_col = f'{prefix}_{i}_hex'
        pct_col = f'{prefix}_{i}_pct'

        if hex_col in plant and plant[hex_col]:
            colors.append({
                'hex': plant[hex_col],
                'percentage': plant[pct_col]
            })

    return colors

def get_dominant_color(plant_id, ranking='frequency'):
    """Get the dominant (first) color for a plant."""
    colors = get_color_palette(plant_id, ranking)
    if colors:
        return colors[0]['hex']
    return None

@st.cache_data
def get_plants_by_color_bucket(bucket_name, ranking='frequency'):
    """Get all plants whose dominant color falls in the given bucket."""
    from .color_utils import categorize_color

    df = load_merged_data()

    # Get dominant color column
    color_col = f'color_freq_1_hex' if ranking == 'frequency' else 'color_visual_1_hex'

    # Filter plants that have a dominant color
    df_with_color = df[df[color_col].notna()].copy()

    # Categorize each plant's dominant color
    df_with_color['color_bucket'] = df_with_color[color_col].apply(categorize_color)

    # Filter by bucket
    return df_with_color[df_with_color['color_bucket'] == bucket_name]

@st.cache_data
def get_color_bucket_stats(ranking='frequency'):
    """Get count of plants in each color bucket."""
    from .color_utils import categorize_color, COLOR_BUCKETS

    df = load_merged_data()

    # Get dominant color column
    color_col = f'color_freq_1_hex' if ranking == 'frequency' else 'color_visual_1_hex'

    # Filter plants that have a dominant color
    df_with_color = df[df[color_col].notna()].copy()

    # Categorize each plant's dominant color
    df_with_color['color_bucket'] = df_with_color[color_col].apply(categorize_color)

    # Count plants per bucket
    counts = df_with_color['color_bucket'].value_counts().to_dict()

    # Return in bucket order, excluding Unknown
    result = {}
    for bucket_name in COLOR_BUCKETS.keys():
        if bucket_name in counts:
            result[bucket_name] = counts[bucket_name]

    return result

def color_distance(hex1, hex2):
    """Calculate Euclidean distance between two hex colors in RGB space."""
    from .color_utils import hex_to_hsv

    try:
        # Convert to RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        rgb1 = hex_to_rgb(hex1)
        rgb2 = hex_to_rgb(hex2)

        # Euclidean distance in RGB space
        return sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5
    except:
        return float('inf')

@st.cache_data
def get_plants_by_color_similarity(target_hex, ranking='frequency', max_distance=50, limit=None):
    """Get plants whose dominant color is similar to the target color.

    Args:
        target_hex: Target hex color (e.g., '#FF5733')
        ranking: 'frequency' or 'visual'
        max_distance: Maximum RGB distance (0-441, lower is more similar)
        limit: Maximum number of plants to return

    Returns:
        DataFrame of plants sorted by color similarity
    """
    df = load_merged_data()

    # Get dominant color column
    color_col = f'color_freq_1_hex' if ranking == 'frequency' else 'color_visual_1_hex'

    # Filter plants that have a dominant color
    df_with_color = df[df[color_col].notna()].copy()

    # Calculate distance from target color
    df_with_color['color_distance'] = df_with_color[color_col].apply(
        lambda x: color_distance(target_hex, x)
    )

    # Filter by max distance and sort
    result = df_with_color[df_with_color['color_distance'] <= max_distance].copy()
    result = result.sort_values('color_distance')

    if limit:
        result = result.head(limit)

    return result
