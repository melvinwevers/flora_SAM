# Flora Batava Plant Explorer

An interactive Streamlit application for exploring 2,241 historical botanical illustrations from the Flora Batava collection.

## Features

- **Browse by Taxonomy**: Navigate through plant families, genera, and species
- **Visual Clusters**: Explore AI-discovered similarity groups using DINOv2, CLIP, PlantNet, and Combined embeddings
- **Color Analysis**: Analyze color patterns and their relationships to taxonomy and clusters

## Quick Start

```bash
# Install dependencies
uv sync

# Run the Streamlit app
uv run streamlit run streamlit_app/Home.py
```

The app will open in your default browser at `http://localhost:8501`.

## Dataset

- **Source**: Flora Batava historical botanical illustrations
- **Plants**: 2,241 segmented specimens
- **Taxonomy**: 473 plants with family/genus data from authoritative Flora Batava spreadsheet
- **Visual Embeddings**: DINOv2 deep learning model
- **Clusters**: 23 visual similarity groups discovered by DINOv2
- **Colors**: Top 5 colors per plant ranked by:
  - **Frequency**: Most common colors by pixel count
  - **Botanical Contrast**: Distinctive features (flowers, fruits, berries) that stand out from green foliage

## Project Structure

```
flora_SAM/
├── streamlit_app/          # Main Streamlit application
│   ├── Home.py            # App entry point
│   ├── pages/             # Individual app pages
│   ├── data/              # Plant metadata and analysis results
│   ├── thumbnails/        # Plant thumbnail images
│   └── utils/             # Shared utilities
├── sam_batch_interactive.py  # SAM segmentation tool (for development)
├── archive/               # Archived development scripts
└── pyproject.toml         # Project dependencies
```

## Data Processing Pipeline

The app uses preprocessed data stored in a single consolidated JSON file (`streamlit_app/data/flora_data.json`). To regenerate or update the data:

### 1. Segment Plant Illustrations

Extract plant segments from illustrations using SAM:

```bash
python sam_batch_interactive.py
```

### 2. Analyze Colors

Extract color features from segmented plant illustrations:

```bash
python color_analysis.py
```

This generates `masks_metadata.json` with plant dimensions and color data.

#### How Botanical Contrast Works

The **Botanical Contrast** ranking identifies distinctive botanical features (flowers, fruits, berries) by measuring how different each color is from typical green foliage.

**Algorithm:**

1. **Reference Palette**: Define 5 typical green foliage colors in LAB space:
   - Dark leaf green: `L=45, a=-20, b=25`
   - Mid leaf green: `L=55, a=-25, b=30`
   - Light leaf green: `L=65, a=-15, b=35`
   - Grayish green: `L=50, a=-5, b=15`
   - Olive green: `L=60, a=-10, b=20`

2. **Distance Calculation**: For each color from k-means clustering (30 clusters), calculate the Euclidean distance in the **a,b plane only** (ignoring lightness L):
   ```
   distance = √[(a₁ - a₂)² + (b₁ - b₂)²]
   ```

   Using only a,b components focuses on hue/chroma differences rather than lightness. This prevents light beige colors from ranking high just because they're lighter than the reference greens.

3. **Contrast Score**: The botanical contrast score is the **minimum distance** to any of the 5 reference greens. Higher distance = more distinctive from foliage.

4. **Filtering**: Remove colors too similar to green foliage (distance < 15). This threshold was chosen to:
   - Filter out most green/olive leaf colors (distance < 15)
   - Keep vivid yellows (distance ~18-20)
   - Keep pinks and reds (distance ~19-25)
   - Keep blues and purples (distance ~20-30)

5. **Ranking**: Sort remaining colors by distance (descending), returning the top 5 most distinctive colors.

**Why This Works:**

In LAB color space, the a-axis ranges from green (negative) to red (positive), and the b-axis ranges from blue (negative) to yellow (positive). By measuring distance from typical greens (negative a, positive b):

- **Yellow flowers**: High positive b, moderate distance (~18-20)
- **Pink/red flowers**: High positive a, good distance (~19-25)
- **Blue flowers**: High negative b, excellent distance (~25-30)
- **Purple flowers**: Positive a + negative b, excellent distance (~20-30)
- **Green leaves**: Negative a + positive b, low distance (< 15, filtered out)

This approach naturally surfaces the distinctive colors that 18th-century botanical illustrators emphasized, without requiring manual hue classification rules.

### 3. Generate DINOv2 Embeddings and Clusters

Extract DINOv2 embeddings and compute HDBSCAN clusters:

```bash
python generate_embeddings.py --input masks/ --use-cache
```

This creates `visualizations/cluster_data/cluster_data_dinov2.json` with cluster assignments.

### 4. Prepare Consolidated Data

Process and consolidate all plant data into a single JSON file:

```bash
python prepare_data.py
```

This creates `streamlit_app/data/flora_data.json` containing:
- Plant colors (frequency and botanical contrast rankings) with full LAB color data
- Authoritative Flora Batava taxonomy from illustration metadata
- DINOv2 cluster assignments (23 visual similarity groups)
- Precomputed comparison metrics


- [Streamlit App Documentation](streamlit_app/README.md)

