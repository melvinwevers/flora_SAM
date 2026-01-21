"""
Generate thumbnails from segmented plant images for Streamlit app.

Resizes *_segmented.png images to 300px wide JPGs for fast web loading.
"""
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm

THUMB_WIDTH = 300
INPUT_DIR = Path("masks/")
OUTPUT_DIR = Path("streamlit_app/thumbnails/")

def generate_thumbnails():
    """Generate thumbnails from all segmented images."""

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all segmented images
    segmented_files = list(INPUT_DIR.glob("*_segmented.png"))

    if not segmented_files:
        print(f"No segmented images found in {INPUT_DIR}")
        return

    print(f"Found {len(segmented_files)} segmented images")
    print(f"Generating {THUMB_WIDTH}px wide thumbnails...")

    success_count = 0
    error_count = 0

    for img_path in tqdm(segmented_files, desc="Generating thumbnails"):
        try:
            # Load image
            img = Image.open(img_path)

            # Calculate new dimensions maintaining aspect ratio
            ratio = THUMB_WIDTH / img.width
            new_height = int(img.height * ratio)

            # Resize with high-quality resampling
            thumb = img.resize((THUMB_WIDTH, new_height), Image.LANCZOS)

            # Convert RGBA to RGB with white background for JPG
            if thumb.mode == 'RGBA':
                background = Image.new('RGB', thumb.size, (255, 255, 255))
                background.paste(thumb, mask=thumb.split()[3])  # Use alpha channel as mask
                thumb = background
            elif thumb.mode != 'RGB':
                thumb = thumb.convert('RGB')

            # Generate output filename
            thumb_name = img_path.stem.replace("_segmented", "") + ".jpg"
            output_path = OUTPUT_DIR / thumb_name

            # Save as JPEG with good quality
            thumb.save(output_path, "JPEG", quality=85, optimize=True)
            success_count += 1

        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            error_count += 1

    print(f"\nComplete!")
    print(f"✓ Successfully generated: {success_count} thumbnails")
    if error_count > 0:
        print(f"✗ Errors: {error_count}")

    # Calculate size savings
    total_size_mb = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.jpg")) / (1024 * 1024)
    print(f"Total thumbnail size: {total_size_mb:.1f} MB")

    avg_size_kb = (total_size_mb * 1024) / success_count if success_count > 0 else 0
    print(f"Average thumbnail size: {avg_size_kb:.1f} KB")

if __name__ == "__main__":
    generate_thumbnails()
