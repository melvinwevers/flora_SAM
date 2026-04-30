#!/usr/bin/env python3
"""
Test different chroma thresholds to see which captures flower colors best.

Compares thresholds: 15 (current), 10, 8, 5
"""
import json
import numpy as np
from pathlib import Path

def analyze_chroma_distribution(masks_meta_path="masks/masks_metadata.json"):
    """Analyze chroma distribution across all plant colors."""

    with open(masks_meta_path) as f:
        masks_meta = json.load(f)

    all_chromas = []

    # Collect all color chroma values
    for plant_id, data in masks_meta.items():
        for ranking in ['colors_frequency', 'colors_saliency', 'colors_chroma']:
            colors = data.get(ranking, [])
            for color in colors:
                if 'lab' in color:
                    lab = color['lab']
                    chroma = np.sqrt(lab[1]**2 + lab[2]**2)
                    all_chromas.append({
                        'plant_id': plant_id,
                        'ranking': ranking,
                        'hex': color.get('hex'),
                        'chroma': chroma,
                        'lab': lab
                    })

    all_chromas.sort(key=lambda x: x['chroma'])

    print("=" * 80)
    print("CHROMA DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"\nTotal colors analyzed: {len(all_chromas)}")

    # Show percentiles
    chroma_values = [c['chroma'] for c in all_chromas]
    percentiles = [5, 10, 25, 50, 75, 90, 95]

    print("\nChroma percentiles:")
    for p in percentiles:
        val = np.percentile(chroma_values, p)
        print(f"  {p:3d}th percentile: {val:6.2f}")

    # Compare different thresholds
    thresholds = [5, 8, 10, 15, 20]
    print("\nPixels retained at different thresholds:")
    for thresh in thresholds:
        retained = sum(1 for c in chroma_values if c > thresh)
        pct = retained / len(chroma_values) * 100
        print(f"  Threshold {thresh:2d}: {retained:6d} colors ({pct:5.1f}%)")

    # Show examples of colors filtered at threshold 15 but kept at 8
    print("\n" + "=" * 80)
    print("COLORS FILTERED AT THRESHOLD 15 (but kept at 8):")
    print("=" * 80)

    filtered_at_15 = [c for c in all_chromas if 8 < c['chroma'] <= 15]

    # Group by plant and show examples
    examples = {}
    for c in filtered_at_15:
        plant_id = c['plant_id']
        if plant_id not in examples:
            examples[plant_id] = []
        examples[plant_id].append(c)

    # Show first 10 plants with filtered colors
    for i, (plant_id, colors) in enumerate(list(examples.items())[:10]):
        print(f"\n{plant_id}:")
        for c in colors[:3]:  # Show up to 3 colors per plant
            print(f"  {c['hex']:8s}  chroma={c['chroma']:5.2f}  LAB({c['lab'][0]:5.1f}, {c['lab'][1]:6.2f}, {c['lab'][2]:6.2f})")

    print("\n" + "=" * 80)
    print(f"Total plants with pastels (chroma 8-15): {len(examples)}")
    print("=" * 80)


if __name__ == "__main__":
    analyze_chroma_distribution()
