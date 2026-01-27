#!/usr/bin/env python3
"""
Export plant color data to analysis-friendly formats.

Creates CSV and JSON exports of color information for all plants,
suitable for external analysis, visualization, or data science workflows.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def export_colors_csv(metadata_file, output_file):
    """
    Export colors to a flat CSV format with one row per color.

    Columns:
    - plant_id: Plant identifier
    - color_rank: 1-5 (rank in frequency order)
    - hex: Hex color code
    - rgb_r, rgb_g, rgb_b: RGB values
    - hsl_h, hsl_s, hsl_l: HSL values
    - lab_l, lab_a, lab_b: LAB values
    - frequency_pct: Area coverage percentage
    - perceptual_weight: Perceptual distinctiveness score
    - saliency_weight: Spatial attention score
    """
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    rows = []

    for mask_key, entry in tqdm(metadata.items(), desc="Exporting colors"):
        plant_id = entry.get('image', mask_key)

        # Remove .tif extension if present
        if plant_id.endswith('.tif'):
            plant_id = plant_id[:-4]

        # Get frequency-ranked colors
        colors_freq = entry.get('colors_frequency', [])

        for rank, color in enumerate(colors_freq, start=1):
            row = {
                'plant_id': plant_id,
                'color_rank': rank,
                'hex': color.get('hex', ''),
                'frequency_pct': color.get('percentage', 0.0),
            }

            # RGB values
            rgb = color.get('rgb', [])
            if len(rgb) == 3:
                row['rgb_r'] = rgb[0]
                row['rgb_g'] = rgb[1]
                row['rgb_b'] = rgb[2]
            else:
                row['rgb_r'] = row['rgb_g'] = row['rgb_b'] = None

            # HSL values
            hsl = color.get('hsl', [])
            if len(hsl) == 3:
                row['hsl_h'] = hsl[0]
                row['hsl_s'] = hsl[1]
                row['hsl_l'] = hsl[2]
            else:
                row['hsl_h'] = row['hsl_s'] = row['hsl_l'] = None

            # LAB values
            lab = color.get('lab', [])
            if len(lab) == 3:
                row['lab_l'] = lab[0]
                row['lab_a'] = lab[1]
                row['lab_b'] = lab[2]
            else:
                row['lab_l'] = row['lab_a'] = row['lab_b'] = None

            # Ranking weights
            row['perceptual_weight'] = color.get('perceptual_weight', None)
            row['saliency_weight'] = color.get('saliency_weight', None)

            rows.append(row)

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

    print(f"\n✓ Exported {len(rows)} colors from {len(metadata)} plants to {output_file}")
    print(f"  Columns: {', '.join(df.columns)}")
    print(f"  File size: {Path(output_file).stat().st_size / 1024:.1f} KB")


def export_colors_json(metadata_file, output_file):
    """
    Export colors to a nested JSON format.

    Structure:
    {
      "plant_id": {
        "colors_frequency": [...],  // Ordered by area coverage
        "colors_perceptual": [...],  // Ordered by perceptual weight
        "colors_saliency": [...]     // Ordered by saliency weight
      }
    }
    """
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    export_data = {}

    for mask_key, entry in tqdm(metadata.items(), desc="Exporting colors"):
        plant_id = entry.get('image', mask_key)

        # Remove .tif extension if present
        if plant_id.endswith('.tif'):
            plant_id = plant_id[:-4]

        # Get all three rankings
        colors_freq = entry.get('colors_frequency', [])
        colors_visual = entry.get('colors_visual', [])

        # Create perceptual and saliency rankings from colors_visual
        if colors_visual and 'perceptual_weight' in colors_visual[0]:
            colors_perceptual = sorted(colors_visual, key=lambda x: x.get('perceptual_weight', 0), reverse=True)
            colors_saliency = sorted(colors_visual, key=lambda x: x.get('saliency_weight', 0), reverse=True)
        else:
            # Fallback to frequency if new format not available
            colors_perceptual = colors_freq
            colors_saliency = colors_freq

        export_data[plant_id] = {
            'colors_frequency': colors_freq[:5],
            'colors_perceptual': colors_perceptual[:5],
            'colors_saliency': colors_saliency[:5]
        }

    # Save JSON
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"\n✓ Exported colors from {len(metadata)} plants to {output_file}")
    print(f"  Format: Nested JSON with 3 rankings per plant")
    print(f"  File size: {Path(output_file).stat().st_size / 1024:.1f} KB")


def export_color_summary(metadata_file, output_file):
    """
    Export a summary CSV with one row per plant and dominant colors.

    Columns:
    - plant_id
    - dominant_color_hex: Most frequent color
    - dominant_color_pct: Coverage percentage
    - salient_color_hex: Most salient color
    - salient_color_pct: Coverage percentage
    - n_colors: Number of colors extracted
    - color_diversity: Shannon entropy of color distribution
    """
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    rows = []

    for mask_key, entry in tqdm(metadata.items(), desc="Creating summary"):
        plant_id = entry.get('image', mask_key)

        # Remove .tif extension if present
        if plant_id.endswith('.tif'):
            plant_id = plant_id[:-4]

        colors_freq = entry.get('colors_frequency', [])
        colors_visual = entry.get('colors_visual', [])

        row = {'plant_id': plant_id}

        # Dominant (most frequent) color
        if colors_freq:
            row['dominant_color_hex'] = colors_freq[0].get('hex', '')
            row['dominant_color_pct'] = colors_freq[0].get('percentage', 0)
        else:
            row['dominant_color_hex'] = ''
            row['dominant_color_pct'] = 0

        # Most salient color
        if colors_visual:
            # Sort by saliency if available
            if 'saliency_weight' in colors_visual[0]:
                most_salient = max(colors_visual, key=lambda x: x.get('saliency_weight', 0))
            else:
                most_salient = colors_visual[0]

            row['salient_color_hex'] = most_salient.get('hex', '')
            row['salient_color_pct'] = most_salient.get('percentage', 0)
        else:
            row['salient_color_hex'] = ''
            row['salient_color_pct'] = 0

        # Color count
        row['n_colors'] = len(colors_freq)

        # Color diversity (Shannon entropy)
        if colors_freq:
            import numpy as np
            percentages = [c.get('percentage', 0) / 100.0 for c in colors_freq]
            percentages = [p for p in percentages if p > 0]
            if percentages:
                entropy = -sum(p * np.log(p) for p in percentages)
                row['color_diversity'] = round(entropy, 4)
            else:
                row['color_diversity'] = 0
        else:
            row['color_diversity'] = 0

        rows.append(row)

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

    print(f"\n✓ Exported summary for {len(metadata)} plants to {output_file}")
    print(f"  Columns: {', '.join(df.columns)}")
    print(f"  File size: {Path(output_file).stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Export plant color data for analysis"
    )
    parser.add_argument(
        "--metadata",
        default="masks/masks_metadata.json",
        help="Input metadata JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="exports",
        help="Output directory for exports"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "summary", "all"],
        default="all",
        help="Export format (default: all)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        return

    print(f"Loading metadata from {metadata_path}...")

    # Export based on format
    if args.format in ["csv", "all"]:
        print("\nExporting detailed CSV...")
        export_colors_csv(
            metadata_path,
            output_dir / "plant_colors_detailed.csv"
        )

    if args.format in ["json", "all"]:
        print("\nExporting nested JSON...")
        export_colors_json(
            metadata_path,
            output_dir / "plant_colors.json"
        )

    if args.format in ["summary", "all"]:
        print("\nExporting summary CSV...")
        export_color_summary(
            metadata_path,
            output_dir / "plant_colors_summary.csv"
        )

    print(f"\n{'='*60}")
    print("EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
