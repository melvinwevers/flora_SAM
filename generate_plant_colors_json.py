#!/usr/bin/env python3
"""
Generate a lightweight JSON file with plant colors for the lite HTML viewer.
Extracts plant names, thumbnail references, and RGB colors from masks_metadata.json.
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Generate plant colors JSON for lite viewer')
    parser.add_argument('--masks-dir', default='masks', help='Directory with masks_metadata.json')
    parser.add_argument('--output', default='visualizations/plant_colors.json', help='Output JSON file')
    parser.add_argument('--max-colors', type=int, default=5, help='Max colors per plant')
    args = parser.parse_args()

    metadata_file = Path(args.masks_dir) / 'masks_metadata.json'
    print(f"Loading metadata from {metadata_file}...")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    plants = []
    for mask_key, entry in metadata.items():
        # Skip entries without color data
        if 'colors_frequency' not in entry or not entry['colors_frequency']:
            continue

        image_name = entry['image']
        # Thumbnail filename matches the image name
        thumb_name = image_name

        colors_freq = entry.get('colors_frequency', [])[:args.max_colors]
        colors_visual = entry.get('colors_visual', colors_freq)[:args.max_colors]

        plants.append({
            'name': image_name.replace('.tif.jpg', '').replace('.jpg', ''),
            'thumb': thumb_name,
            'colors_freq': [
                {'rgb': c['rgb'], 'pct': c['percentage']}
                for c in colors_freq
            ],
            'colors_visual': [
                {'rgb': c['rgb'], 'pct': c['percentage']}
                for c in colors_visual
            ]
        })

    # Sort by name for consistent ordering
    plants.sort(key=lambda p: p['name'])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(plants, f)

    file_size = output_path.stat().st_size / 1024
    print(f"Generated {output_path} ({len(plants)} plants, {file_size:.1f} KB)")


if __name__ == '__main__':
    main()
