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
    df = pd.read_csv(DATA_DIR / "flora_batava_index.csv")
    return df

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
        'visual': 'color_visual'  # backwards compatibility
    }

    prefix = prefix_map.get(ranking, 'color_freq')

    # Fallback for missing columns (until data regeneration)
    # Check if first column exists, if not fall back to color_visual
    test_col = f'{prefix}_1_hex'
    if test_col not in plant:
        prefix = 'color_visual'

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

def hex_to_lab(hex_color: str):
    """Convert hex color to LAB color space.

    LAB is perceptually uniform - equal distances represent equal
    perceived color differences.

    Args:
        hex_color: Hex color string (e.g., '#FF5733')

    Returns:
        Tuple of (L, a, b) where:
        - L: Lightness (0-100)
        - a: Green-Red axis (-128 to 127)
        - b: Blue-Yellow axis (-128 to 127)
    """
    from skimage.color import rgb2lab
    import numpy as np

    # Convert hex to RGB (normalize to 0-1 range)
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0

    # Convert to LAB using scikit-image
    rgb_normalized = np.array([[[r, g, b]]])
    lab = rgb2lab(rgb_normalized)

    # Extract LAB values
    l_value = float(lab[0, 0, 0])
    a_value = float(lab[0, 0, 1])
    b_value = float(lab[0, 0, 2])

    return (l_value, a_value, b_value)

def color_distance(hex1, hex2):
    """Calculate Delta E (CIE76) distance between two hex colors in LAB space.

    Delta E is perceptually uniform - equal distances represent equal
    perceived color differences. This is the standard color difference metric
    in color science.

    Delta E Scale:
    - 0-1:     Imperceptible difference
    - 1-2:     Just noticeable difference (JND)
    - 2-10:    Noticeable difference
    - 10-20:   Very different
    - 20+:     Completely different

    Args:
        hex1: First hex color (e.g., '#FF5733')
        hex2: Second hex color (e.g., '#33FF57')

    Returns:
        Delta E distance (0-150 typical range, 0-200 max)
    """
    try:
        lab1 = hex_to_lab(hex1)
        lab2 = hex_to_lab(hex2)

        # Calculate Delta E (CIE76) - Euclidean distance in LAB space
        delta_L = lab1[0] - lab2[0]
        delta_a = lab1[1] - lab2[1]
        delta_b = lab1[2] - lab2[2]

        delta_e = (delta_L**2 + delta_a**2 + delta_b**2) ** 0.5

        return delta_e
    except Exception as e:
        print(f"Error calculating color distance: {e}")
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

def get_family_color_palette(family, ranking='frequency', top_n=8):
    """Get dominant colors for a family, binned by hue.

    Args:
        family: Family name
        ranking: Color ranking to use ('frequency' or 'saliency')
        top_n: Maximum number of colors to return

    Returns:
        List of dicts with 'hex', 'bucket', and 'count' keys
    """
    from utils.color_utils import get_group_color_palette

    df = load_merged_data()
    family_df = df[df['family'] == family]

    # Get all dominant color hexes for this family
    color_col = f'color_freq_1_hex' if ranking == 'frequency' else 'color_visual_1_hex'
    colors = family_df[color_col].dropna().tolist()

    return get_group_color_palette(colors, top_n=top_n)


def get_genus_color_palette(genus, ranking='frequency', top_n=8):
    """Get dominant colors for a genus, binned by hue.

    Args:
        genus: Genus name
        ranking: Color ranking to use ('frequency' or 'saliency')
        top_n: Maximum number of colors to return

    Returns:
        List of dicts with 'hex', 'bucket', and 'count' keys
    """
    from utils.color_utils import get_group_color_palette

    df = load_merged_data()
    genus_df = df[df['genus'] == genus]

    # Get all dominant color hexes for this genus
    color_col = f'color_freq_1_hex' if ranking == 'frequency' else 'color_visual_1_hex'
    colors = genus_df[color_col].dropna().tolist()

    return get_group_color_palette(colors, top_n=top_n)


def get_cluster_color_palette(model, cluster_id, ranking='frequency', top_n=8):
    """Get dominant colors for a cluster, binned by hue.

    Args:
        model: Clustering model name (e.g., 'dinov2')
        cluster_id: Cluster ID
        ranking: Color ranking to use ('frequency' or 'saliency')
        top_n: Maximum number of colors to return

    Returns:
        List of dicts with 'hex', 'bucket', and 'count' keys
    """
    from utils.color_utils import get_group_color_palette

    df = load_merged_data()

    if model not in df.columns:
        return []

    cluster_df = df[df[model] == cluster_id]

    # Get all dominant color hexes for this cluster
    color_col = f'color_freq_1_hex' if ranking == 'frequency' else 'color_visual_1_hex'
    colors = cluster_df[color_col].dropna().tolist()

    return get_group_color_palette(colors, top_n=top_n)
