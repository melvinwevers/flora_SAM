# Flora SAM

Interactive segmentation tool for botanical illustrations using Meta's Segment Anything Model (SAM).

## Quick Start

```bash
# Install dependencies
uv sync

# Run interactive segmentation
python sam_batch_interactive.py
```

The script automatically downloads the SAM model on first run (~375MB).

## Usage

Process botanical illustrations from the Flora Batava dataset:

- **Click**: Add positive point (include area)
- **Shift+Click**: Add negative point (exclude area)
- **z**: Undo last point
- **s**: Save mask and move to next image
- **r**: Reset current image
- **u**: Skip to next unprocessed image
- **q**: Quit

## Output

The tool generates:
- Binary masks (`*_mask.png`)
- Segmented images (`*_segmented.png`)
- JSON metadata file with mask areas and image dimensions

All outputs are saved to the `masks/` directory.

## Configuration

```bash
python sam_batch_interactive.py \
  --data-dir data \
  --meta flora_meta.xlsx \
  --model sam_vit_b.pth \
  --output masks
```
