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
def load_plants_metadata():
    """Load plants metadata CSV with colors and taxonomy."""
    df = pd.read_csv(DATA_DIR / "plants_metadata.csv")
    return df

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
    # Thumbnails are named without .tif extension
    return THUMBNAILS_DIR / f"{plant_id}.jpg"

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
    """Get color palette for a plant."""
    plant = get_plant_details(plant_id)

    if not plant:
        return []

    prefix = 'color_freq' if ranking == 'frequency' else 'color_visual'

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
