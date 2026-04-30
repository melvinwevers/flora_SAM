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
  - Frequency (pixel count)
  - Saliency (visual attention)
  - Chroma (color vividness)

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

This generates `masks_metadata.json` with plant dimensions and color data (frequency, saliency, chroma rankings).

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
- Plant colors (frequency, saliency, chroma rankings) with full LAB color data
- Authoritative Flora Batava taxonomy from illustration metadata
- DINOv2 cluster assignments (23 visual similarity groups)
- Precomputed comparison metrics


- [Streamlit App Documentation](streamlit_app/README.md)

