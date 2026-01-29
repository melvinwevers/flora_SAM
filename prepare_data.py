"""
Prepare data files for Streamlit app from existing analysis outputs.

This script:
1. Creates slimmed metadata CSV from masks_metadata.json
2. Extracts UMAP coordinates and cluster assignments
3. Precomputes comparison metrics (contingency, purity, surprising plants)
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Paths
MASKS_META = Path("masks/masks_metadata.json")
COLOR_FEATURES = Path("visualizations/color_features.csv")
CLUSTER_DATA_DIR = Path("visualizations/cluster_data")
OUTPUT_DIR = Path("streamlit_app/data")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_masks_metadata():
    """Load masks metadata JSON."""
    print("Loading masks metadata...")
    with open(MASKS_META) as f:
        return json.load(f)

def create_plants_metadata_csv(masks_meta):
    """Create slimmed CSV with essential plant metadata."""
    print("Creating plants_metadata.csv...")

    records = []
    for mask_key, entry in masks_meta.items():
        # Extract filename without path
        filename = Path(entry['segmented_file']).name
        plant_id = filename.replace('_segmented.png', '')

        # Remove .tif extension if present to match cluster data
        if plant_id.endswith('.tif'):
            plant_id = plant_id[:-4]

        # Extract top 5 colors from all three rankings
        colors_freq = entry.get('colors_frequency', [])[:5]
        colors_perceptual = entry.get('colors_perceptual', [])[:5]
        colors_saliency = entry.get('colors_saliency', [])[:5]
        colors_visual = entry.get('colors_visual', [])[:5]  # backwards compatibility

        record = {
            'plant_id': plant_id,
            'filename': filename,
            'area_pixels': entry.get('area_pixels', 0),
            'width': entry.get('image_dimensions', {}).get('width', 0),
            'height': entry.get('image_dimensions', {}).get('height', 0),
        }

        # Add frequency-ranked colors
        for i, color in enumerate(colors_freq, 1):
            record[f'color_freq_{i}_hex'] = color.get('hex', '')
            record[f'color_freq_{i}_pct'] = color.get('percentage', 0)

        # Add perceptual-ranked colors
        for i, color in enumerate(colors_perceptual, 1):
            record[f'color_perceptual_{i}_hex'] = color.get('hex', '')
            record[f'color_perceptual_{i}_pct'] = color.get('percentage', 0)

        # Add saliency-ranked colors
        for i, color in enumerate(colors_saliency, 1):
            record[f'color_saliency_{i}_hex'] = color.get('hex', '')
            record[f'color_saliency_{i}_pct'] = color.get('percentage', 0)

        # Add visual importance colors (backwards compatibility)
        for i, color in enumerate(colors_visual, 1):
            record[f'color_visual_{i}_hex'] = color.get('hex', '')
            record[f'color_visual_{i}_pct'] = color.get('percentage', 0)

        records.append(record)

    df = pd.DataFrame(records)
    output_path = OUTPUT_DIR / "plants_metadata.csv"
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(df)} plants to {output_path}")
    return df

def extract_cluster_assignments():
    """Extract cluster assignments from cluster data JSONs."""
    print("Extracting cluster assignments...")

    models = ['dinov2', 'clip', 'plantnet', 'combined']
    all_assignments = []

    for model in models:
        json_path = CLUSTER_DATA_DIR / f"cluster_data_{model}.json"

        if not json_path.exists():
            print(f"  ⚠ Warning: {json_path} not found, skipping {model}")
            continue

        with open(json_path) as f:
            cluster_data = json.load(f)

        # Extract plant_id → cluster mapping
        for cluster_id, plants in cluster_data.items():
            for plant in plants:
                # Extract base filename without extension
                plant_id = plant['plant_id']
                # Remove common extensions
                for ext in ['.tif.jpg', '.jpg', '.tif']:
                    if plant_id.endswith(ext):
                        plant_id = plant_id[:-len(ext)]
                        break

                all_assignments.append({
                    'plant_id': plant_id,
                    'model': model,
                    'cluster': int(cluster_id) if cluster_id != '-1' else -1
                })

    df = pd.DataFrame(all_assignments)

    # Pivot to wide format: one row per plant, one column per model
    df_wide = df.pivot(index='plant_id', columns='model', values='cluster').reset_index()
    df_wide.columns.name = None

    output_path = OUTPUT_DIR / "cluster_assignments.csv"
    df_wide.to_csv(output_path, index=False)
    print(f"  ✓ Saved cluster assignments to {output_path}")
    return df_wide

def link_taxonomic_data(plants_df):
    """Link plants with taxonomic data from color_features.csv."""
    print("Linking taxonomic data...")

    if not COLOR_FEATURES.exists():
        print(f"  ⚠ Warning: {COLOR_FEATURES} not found, skipping taxonomy")
        return plants_df

    color_df = pd.read_csv(COLOR_FEATURES)

    # Extract plant_id from mask_file column, remove .tif extension to match
    color_df['plant_id'] = color_df['mask_file'].str.replace('_mask.png', '').str.replace('.tif', '')

    # Select taxonomic columns
    tax_cols = ['plant_id', 'family', 'genus', 'species']
    tax_cols = [c for c in tax_cols if c in color_df.columns]

    if len(tax_cols) > 1:
        taxonomy = color_df[tax_cols].drop_duplicates()
        plants_df = plants_df.merge(taxonomy, on='plant_id', how='left')
        print(f"  ✓ Linked {plants_df['family'].notna().sum()} plants with taxonomy")

    return plants_df

def compute_comparison_metrics(plants_df, clusters_df):
    """Precompute taxonomy vs cluster comparison metrics."""
    print("Computing comparison metrics...")

    # Merge data
    df = plants_df.merge(clusters_df, on='plant_id', how='inner')

    # Check if taxonomy data exists
    if 'family' not in df.columns:
        print("  ⚠ Warning: No family column, skipping comparison metrics")
        return None

    n_with_family = df['family'].notna().sum()
    if n_with_family == 0:
        print("  ⚠ Warning: No taxonomy data available, skipping comparison metrics")
        return None

    print(f"  Found {n_with_family} plants with taxonomy data")

    metrics = {}

    # For each model
    for model in ['dinov2', 'clip', 'plantnet', 'combined']:
        if model not in df.columns:
            continue

        # Filter out noise (-1) and missing data
        valid = df[(df[model] >= 0) & df['family'].notna()].copy()

        if len(valid) == 0:
            continue

        model_metrics = {}

        # Overall agreement scores
        model_metrics['adjusted_rand_index'] = float(
            adjusted_rand_score(valid['family'], valid[model])
        )
        model_metrics['normalized_mutual_info'] = float(
            normalized_mutual_info_score(valid['family'], valid[model])
        )

        # Contingency matrix (family × cluster)
        contingency = pd.crosstab(valid['family'], valid[model])
        model_metrics['contingency_matrix'] = contingency.to_dict()

        # Cluster purity: % of dominant family in each cluster
        cluster_purity = valid.groupby(model)['family'].apply(
            lambda x: (x.value_counts().iloc[0] / len(x)) if len(x) > 0 else 0
        ).to_dict()
        model_metrics['cluster_purity'] = {int(k): float(v) for k, v in cluster_purity.items()}

        # Family concentration: % of family in dominant cluster
        family_concentration = valid.groupby('family')[model].apply(
            lambda x: (x.value_counts().iloc[0] / len(x)) if len(x) > 0 else 0
        ).to_dict()
        model_metrics['family_concentration'] = {str(k): float(v) for k, v in family_concentration.items()}

        # Surprising plants: same cluster, different family
        surprising = []
        for cluster_id in valid[model].unique():
            cluster_plants = valid[valid[model] == cluster_id]
            if len(cluster_plants) < 2:
                continue

            # Find dominant family
            dominant_family = cluster_plants['family'].mode()[0]

            # Find plants not in dominant family
            outliers = cluster_plants[cluster_plants['family'] != dominant_family]

            for _, plant in outliers.head(20).iterrows():  # Limit to top 20 per cluster
                surprising.append({
                    'plant_id': plant['plant_id'],
                    'family': plant['family'],
                    'cluster': int(cluster_id),
                    'dominant_family': dominant_family,
                    'cluster_size': len(cluster_plants)
                })

        model_metrics['surprising_plants'] = surprising[:100]  # Top 100 overall

        metrics[model] = model_metrics

    # Save metrics
    output_path = OUTPUT_DIR / "comparison_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"  ✓ Saved comparison metrics to {output_path}")
    return metrics

def main():
    """Run all data preparation steps."""
    print("=" * 60)
    print("Preparing data for Streamlit app")
    print("=" * 60)

    # Step 1: Load masks metadata
    masks_meta = load_masks_metadata()

    # Step 2: Create plants metadata CSV
    plants_df = create_plants_metadata_csv(masks_meta)

    # Step 3: Link taxonomic data
    plants_df = link_taxonomic_data(plants_df)

    # Save updated plants metadata with taxonomy
    output_path = OUTPUT_DIR / "plants_metadata.csv"
    plants_df.to_csv(output_path, index=False)

    # Step 4: Extract cluster assignments
    clusters_df = extract_cluster_assignments()

    # Step 5: Compute comparison metrics
    if len(clusters_df) > 0:
        metrics = compute_comparison_metrics(plants_df, clusters_df)

    # Step 6: Convert Flora Batava Index Excel to CSV (removes Excel dependency)
    flora_index_excel = Path("data/Flora Batava Index.xlsx")
    if flora_index_excel.exists():
        print("Converting Flora Batava Index to CSV...")
        flora_df = pd.read_excel(flora_index_excel)
        flora_subset = flora_df[['Huidige Nederlandse naam', 'Oude Nederlandse naam in Flora Batava',
                                  'Huidige botanische naam', 'Link naar scans in bladerboek KB']]
        flora_csv_path = OUTPUT_DIR / "flora_batava_index.csv"
        flora_subset.to_csv(flora_csv_path, index=False)
        print(f"  ✓ Saved {len(flora_subset)} entries to {flora_csv_path}")
    else:
        print(f"  ⚠ Warning: {flora_index_excel} not found, skipping Flora Batava Index conversion")

    print("=" * 60)
    print("Data preparation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    # Summary
    print(f"\nFiles created:")
    for f in OUTPUT_DIR.glob("*"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name}: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
