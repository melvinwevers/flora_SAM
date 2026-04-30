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

def create_plants_metadata_json(masks_meta):
    """Create JSON with structured plant metadata including full color data."""
    print("Creating plants_metadata.json...")

    plants = {}
    for mask_key, entry in masks_meta.items():
        # Extract filename without path
        filename = Path(entry['segmented_file']).name
        plant_id = filename.replace('_segmented.png', '')

        # Remove .tif extension if present to match cluster data
        if plant_id.endswith('.tif'):
            plant_id = plant_id[:-4]

        # Extract top 5 colors from frequency and vividness (chroma) rankings
        colors_freq = entry.get('colors_frequency', [])[:5]
        colors_chroma = entry.get('colors_chroma', [])[:5]

        # Calculate mask dimensions and percentage of full image
        mask_width = entry.get('image_dimensions', {}).get('width', 0)
        mask_height = entry.get('image_dimensions', {}).get('height', 0)
        mask_area = entry.get('area_pixels', 0)

        # Calculate percentage of mask relative to full image dimensions
        total_mask_box_pixels = mask_width * mask_height if mask_width and mask_height else 0
        mask_percentage = (mask_area / total_mask_box_pixels * 100) if total_mask_box_pixels > 0 else 0

        # Structure colors as list of dictionaries with full LAB data
        frequency_colors = []
        for color in colors_freq:
            lab = color.get('lab', [0, 0, 0])
            rgb = color.get('rgb', [0, 0, 0])
            frequency_colors.append({
                'hex': color.get('hex', ''),
                'percentage': round(color.get('percentage', 0), 2),
                'rgb': rgb,
                'lab': {
                    'L': round(lab[0], 2),
                    'a': round(lab[1], 2),
                    'b': round(lab[2], 2)
                }
            })

        vividness_colors = []
        for color in colors_chroma:
            lab = color.get('lab', [0, 0, 0])
            rgb = color.get('rgb', [0, 0, 0])
            chroma = color.get('chroma', 0)
            vividness_colors.append({
                'hex': color.get('hex', ''),
                'percentage': round(color.get('percentage', 0), 2),
                'chroma': round(chroma, 2),
                'rgb': rgb,
                'lab': {
                    'L': round(lab[0], 2),
                    'a': round(lab[1], 2),
                    'b': round(lab[2], 2)
                }
            })

        plants[plant_id] = {
            'plant_id': plant_id,
            'filename': filename,
            'dimensions': {
                'width': mask_width,
                'height': mask_height,
                'area_pixels': mask_area,
                'mask_percentage': round(mask_percentage, 2)
            },
            'colors': {
                'frequency': frequency_colors,
                'vividness': vividness_colors
            }
        }

    output_path = OUTPUT_DIR / "plants_metadata.json"
    with open(output_path, 'w') as f:
        json.dump(plants, f, indent=2)
    print(f"  ✓ Saved {len(plants)} plants to {output_path}")
    return plants

def extract_cluster_assignments():
    """Extract cluster assignments from cluster data JSONs."""
    print("Extracting cluster assignments...")

    models = ['dinov2']
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

def create_consolidated_json(masks_meta):
    """
    Create consolidated JSON with all plant data, taxonomy, clusters, and metrics.

    This single JSON replaces multiple CSV files:
    - plants_metadata.csv
    - flora_metadata.csv
    - cluster_assignments.csv
    - comparison_metrics.json
    """
    print("Creating consolidated flora_data.json...")
    import csv as csv_mod
    from collections import defaultdict

    # Step 1: Build base plant data from masks metadata
    plants = {}
    for mask_key, entry in masks_meta.items():
        filename = Path(entry['segmented_file']).name
        plant_id = filename.replace('_segmented.png', '')
        if plant_id.endswith('.tif'):
            plant_id = plant_id[:-4]

        # Extract colors (frequency, saliency, chroma)
        colors_freq = entry.get('colors_frequency', [])[:5]
        colors_saliency = entry.get('colors_saliency', [])[:5]
        colors_chroma = entry.get('colors_chroma', [])[:5]

        # Dimensions
        mask_width = entry.get('image_dimensions', {}).get('width', 0)
        mask_height = entry.get('image_dimensions', {}).get('height', 0)
        mask_area = entry.get('area_pixels', 0)
        total_pixels = mask_width * mask_height if mask_width and mask_height else 0
        mask_percentage = (mask_area / total_pixels * 100) if total_pixels > 0 else 0

        # Helper to format color
        def format_color(color):
            lab = color.get('lab', [0, 0, 0])
            return {
                'hex': color.get('hex', ''),
                'percentage': round(color.get('percentage', 0), 2),
                'rgb': color.get('rgb', [0, 0, 0]),
                'lab': {'L': round(lab[0], 2), 'a': round(lab[1], 2), 'b': round(lab[2], 2)}
            }

        def format_chroma_color(color):
            result = format_color(color)
            result['chroma'] = round(color.get('chroma', 0), 2)
            return result

        plants[plant_id] = {
            'plant_id': plant_id,
            'filename': filename,
            'dimensions': {
                'width': mask_width,
                'height': mask_height,
                'area_pixels': mask_area,
                'mask_percentage': round(mask_percentage, 2)
            },
            'colors': {
                'frequency': [format_color(c) for c in colors_freq],
                'saliency': [format_color(c) for c in colors_saliency],
                'chroma': [format_chroma_color(c) for c in colors_chroma]
            },
            'taxonomy': {},  # Will be filled in step 2
            'clusters': {}   # Will be filled in step 3
        }

    # Step 2: Add taxonomy from Flora Batava illustration file names
    illustration_csv = Path("streamlit_app/data/Flora Batava - illustration file names final.csv")
    if illustration_csv.exists():
        print("  Merging taxonomy from Flora Batava illustration spreadsheet...")
        with open(illustration_csv, encoding='utf-8-sig') as f:
            reader = csv_mod.DictReader(f, delimiter=';')
            ill_rows = [r for r in reader if r['File name'].strip()]

        file_to_rows = defaultdict(list)
        for r in ill_rows:
            fid = r['File name'].replace('.tif.jpg', '')
            file_to_rows[fid].append(r)

        for fid, file_rows in file_to_rows.items():
            if fid not in plants:
                continue  # Skip plants without mask data

            if len(file_rows) == 1:
                r = file_rows[0]
                plants[fid]['taxonomy'] = {
                    'volume': r['Volume'],
                    'species_raw': r['Species name RAW'],
                    'species_current': r['Species name CURRENT'],
                    'genus': r['Genus'],
                    'epithet': r['Epithet'],
                    'group': r['Group'],
                    'family': r['Family'],
                    'dutch_name': r['Dutch common name'].strip() or None,
                    'kb_link': r['Link.KB'].strip() or None,
                    'multi_species': False,
                    'all_species': r['Species name CURRENT'],
                }
            else:
                species_list = [r['Species name CURRENT'] for r in file_rows]
                dutch_names = [r['Dutch common name'].strip() for r in file_rows if r['Dutch common name'].strip()]
                families = list(dict.fromkeys(r['Family'] for r in file_rows))
                genera = list(dict.fromkeys(r['Genus'] for r in file_rows))
                r0 = file_rows[0]
                plants[fid]['taxonomy'] = {
                    'volume': r0['Volume'],
                    'species_raw': ' / '.join(r['Species name RAW'] for r in file_rows),
                    'species_current': ' / '.join(species_list),
                    'genus': ' / '.join(genera),
                    'epithet': ' / '.join(r['Epithet'] for r in file_rows),
                    'group': r0['Group'],
                    'family': ' / '.join(families),
                    'dutch_name': ' / '.join(dutch_names) if dutch_names else None,
                    'kb_link': r0['Link.KB'].strip() or None,
                    'multi_species': True,
                    'all_species': ' / '.join(species_list),
                }

    # Step 3: Add cluster assignments
    models = ['dinov2']
    for model in models:
        json_path = CLUSTER_DATA_DIR / f"cluster_data_{model}.json"
        if not json_path.exists():
            continue

        with open(json_path) as f:
            cluster_data = json.load(f)

        for cluster_id_str, cluster_plants in cluster_data.items():
            for plant in cluster_plants:
                plant_id = plant['plant_id']
                for ext in ['.tif.jpg', '.jpg', '.tif']:
                    if plant_id.endswith(ext):
                        plant_id = plant_id[:-len(ext)]
                        break

                if plant_id in plants:
                    plants[plant_id]['clusters'][model] = int(cluster_id_str) if cluster_id_str != '-1' else -1

    # Step 4: Compute comparison metrics
    print("  Computing comparison metrics...")
    # Build temporary dataframe for metrics computation
    records = []
    for plant_id, plant in plants.items():
        if not plant['taxonomy'] or 'family' not in plant['taxonomy']:
            continue
        record = {
            'plant_id': plant_id,
            'family': plant['taxonomy'].get('family')
        }
        for model in models:
            record[model] = plant['clusters'].get(model, -1)
        records.append(record)

    df = pd.DataFrame(records)
    metrics = {}

    for model in models:
        if model not in df.columns:
            continue

        valid = df[(df[model] >= 0) & df['family'].notna()].copy()
        if len(valid) == 0:
            continue

        model_metrics = {
            'adjusted_rand_index': float(adjusted_rand_score(valid['family'], valid[model])),
            'normalized_mutual_info': float(normalized_mutual_info_score(valid['family'], valid[model])),
            'contingency_matrix': pd.crosstab(valid['family'], valid[model]).to_dict(),
            'cluster_purity': {
                int(k): float(v) for k, v in
                valid.groupby(model)['family'].apply(
                    lambda x: (x.value_counts().iloc[0] / len(x)) if len(x) > 0 else 0
                ).to_dict().items()
            },
            'family_concentration': {
                str(k): float(v) for k, v in
                valid.groupby('family')[model].apply(
                    lambda x: (x.value_counts().iloc[0] / len(x)) if len(x) > 0 else 0
                ).to_dict().items()
            }
        }

        # Surprising plants
        surprising = []
        for cluster_id in valid[model].unique():
            cluster_plants = valid[valid[model] == cluster_id]
            if len(cluster_plants) < 2:
                continue
            dominant_family = cluster_plants['family'].mode()[0]
            outliers = cluster_plants[cluster_plants['family'] != dominant_family]
            for _, plant in outliers.head(20).iterrows():
                surprising.append({
                    'plant_id': plant['plant_id'],
                    'family': plant['family'],
                    'cluster': int(cluster_id),
                    'dominant_family': dominant_family,
                    'cluster_size': len(cluster_plants)
                })
        model_metrics['surprising_plants'] = surprising[:100]
        metrics[model] = model_metrics

    # Step 5: Create final consolidated structure
    consolidated = {
        'plants': plants,
        'comparison_metrics': metrics
    }

    # Save consolidated JSON
    output_path = OUTPUT_DIR / "flora_data.json"
    with open(output_path, 'w') as f:
        json.dump(consolidated, f, indent=2)

    print(f"  ✓ Saved consolidated data to {output_path}")
    print(f"  ✓ {len(plants)} plants with taxonomy, colors, and clusters")
    return consolidated

def main():
    """Run all data preparation steps."""
    print("=" * 60)
    print("Preparing data for Streamlit app")
    print("=" * 60)

    # Load masks metadata
    masks_meta = load_masks_metadata()

    # Create consolidated JSON with all data (plants, taxonomy, clusters, metrics)
    consolidated = create_consolidated_json(masks_meta)

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
