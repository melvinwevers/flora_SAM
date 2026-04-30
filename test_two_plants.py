#!/usr/bin/env python3
"""
Test color analysis on two specific plants to verify chroma threshold improvement.
"""
import json
import sys
from pathlib import Path

# Add color_analysis functions
sys.path.insert(0, str(Path(__file__).parent))
from color_analysis import ColorAnalyzer

def test_specific_plants():
    """Test color analysis on KW_T_423_1_0020R and KW_T_423_12_0024L"""

    test_plants = [
        "KW_T_423_1_0020R.tif_mask.png",  # Should have blue flowers
        "KW_T_423_12_0024L.tif_mask.png"  # Should have pink
    ]

    print("=" * 80)
    print("TESTING CHROMA THRESHOLD CHANGE (15 → 8)")
    print("=" * 80)
    print("\nTarget plants:")
    for plant in test_plants:
        print(f"  - {plant}")

    # Initialize analyzer
    analyzer = ColorAnalyzer(
        mask_output_dir="masks/",
        n_colors=10,
        force=True,  # Force reprocessing
        n_jobs=1
    )

    # Load existing metadata
    with open("masks/masks_metadata.json") as f:
        metadata = json.load(f)

    print("\n" + "=" * 80)
    print("PROCESSING WITH NEW THRESHOLD (chroma > 8)")
    print("=" * 80)

    for plant_key in test_plants:
        if plant_key not in metadata:
            print(f"\nWarning: {plant_key} not found in metadata")
            continue

        entry = metadata[plant_key]
        segmented_file = entry.get('segmented_file')

        if not segmented_file:
            print(f"\nWarning: No segmented_file for {plant_key}")
            continue

        segmented_path = Path("masks") / segmented_file
        if not segmented_path.exists():
            print(f"\nWarning: {segmented_path} does not exist")
            continue

        print(f"\n{'='*80}")
        print(f"Plant: {plant_key}")
        print(f"{'='*80}")

        # Import worker function
        from color_analysis import _process_single_image

        background_color = entry.get('background_color_rgb')
        if background_color and isinstance(background_color, list):
            background_color = tuple(background_color)

        work_item = (
            plant_key,
            segmented_path,
            background_color,
            10,  # n_colors
            True,  # filter_background
            50,  # background_tolerance
            True,  # crop_borders
        )

        # Process single image
        _, color_results, error = _process_single_image(work_item)

        if error:
            print(f"Error: {error}")
            continue

        # Display results
        print("\nCHROMA RANKING (sorted by vividness):")
        print("-" * 80)
        for i, color in enumerate(color_results.get('chroma', [])[:5], 1):
            hex_code = color.get('hex', '???')
            chroma = color.get('chroma', 0)
            pct = color.get('percentage', 0)
            lab = color.get('lab', [0, 0, 0])

            print(f"{i}. {hex_code:8s}  chroma={chroma:5.2f}  {pct:5.2f}%  "
                  f"LAB(L={lab[0]:5.1f}, a={lab[1]:6.2f}, b={lab[2]:6.2f})")

        print("\nFREQUENCY RANKING (sorted by area):")
        print("-" * 80)
        for i, color in enumerate(color_results.get('frequency', [])[:5], 1):
            hex_code = color.get('hex', '???')
            pct = color.get('percentage', 0)
            lab = color.get('lab', [0, 0, 0])

            print(f"{i}. {hex_code:8s}  {pct:5.2f}%  "
                  f"LAB(L={lab[0]:5.1f}, a={lab[1]:6.2f}, b={lab[2]:6.2f})")

        print("\nSALIENCY RANKING (sorted by visual attention):")
        print("-" * 80)
        for i, color in enumerate(color_results.get('saliency', [])[:5], 1):
            hex_code = color.get('hex', '???')
            pct = color.get('percentage', 0)
            lab = color.get('lab', [0, 0, 0])

            print(f"{i}. {hex_code:8s}  {pct:5.2f}%  "
                  f"LAB(L={lab[0]:5.1f}, a={lab[1]:6.2f}, b={lab[2]:6.2f})")

    print("\n" + "=" * 80)
    print("DONE - Check if blue/pink flower colors now appear in chroma ranking!")
    print("=" * 80)

if __name__ == "__main__":
    test_specific_plants()
