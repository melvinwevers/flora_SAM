# Plant UMAP Visualization Design

**Date:** 2026-01-21
**Goal:** Create an interactive UMAP visualization tool for 2,243 botanical illustrations using multiple embeddings (DINOv2, CLIP, PlantNet) to explore botanical diversity and identify segmentation outliers.

## Overview

A Python-based pipeline that:
1. Extracts embeddings from plant images using three complementary models
2. Reduces dimensionality with UMAP for visualization
3. Detects outliers using multiple methods
4. Creates interactive Plotly dashboards for exploration
5. Caches all computations for fast iteration

## Architecture

### File Structure
```
plant_visualization.py          # Main script
embeddings_cache/               # Cached embeddings directory
  ├── dinov2_embeddings.npz
  ├── clip_embeddings.npz
  ├── plantnet_embeddings.npz
  └── umap_projections.npz      # Cached UMAP results
visualizations/                 # Output directory
  ├── umap_interactive.html     # Interactive Plotly dashboard
  ├── outliers_report.html      # Outlier analysis report
  └── cluster_analysis.json     # Statistics and metadata
masks/                          # Input: 2,243 segmented plant images
```

### Core Components

**1. EmbeddingExtractor class**
- Loads and manages three models: DINOv2, CLIP, PlantNet
- Extracts features with GPU acceleration (24GB VRAM available)
- Implements caching and checkpoint resume functionality
- Batch processing for efficiency

**2. UMAPVisualizer class**
- Computes UMAP projections for each embedding type
- Creates combined embedding projection
- Generates interactive Plotly visualizations
- Handles caching of UMAP results

**3. OutlierDetector class**
- Multiple detection methods (LOF, disagreement, distance, density)
- Confidence scoring (high/medium/low)
- Generates outlier reports with thumbnails

**4. main() function**
- Orchestrates pipeline with command-line options
- Progress reporting
- Error handling and resume capability

## Embedding Strategy

### Model Selection

**DINOv2 Large** (`facebook/dinov2-large`)
- 1024-dimensional embeddings
- Self-supervised vision transformer
- Excellent for morphological features (shape, structure, spatial relationships)
- Pure visual understanding without text

**CLIP Large** (`openai/clip-vit-large-patch14`)
- 768-dimensional embeddings
- Vision-language model
- Captures semantic concepts and general botanical features
- Complements DINOv2's visual approach

**PlantNet** (largest available plant-specific model from timm/HuggingFace)
- Model size TBD (likely `vit_large_patch14_224` fine-tuned on PlantCLEF)
- Trained specifically on plant images
- Captures taxonomically-relevant discriminative features

### Extraction Process

```python
class EmbeddingExtractor:
    def __init__(self):
        # Load all 3 models at startup (~8-12GB total VRAM)
        self.dinov2_model = load_dinov2_large()
        self.clip_model = load_clip_large()
        self.plantnet_model = load_plantnet()

    def extract_all(self, image_paths):
        # Process in batches of 32-64 (auto-detect optimal size)
        # Process each image through all 3 models in one pass
        # Checkpoint every 100 images
        # Expected time: 10-15 minutes for 2,243 images
```

**Preprocessing:**
- Resize images to 224x224 (model input size)
- Normalize according to each model's requirements
- Handle grayscale/RGBA conversion as needed

**Caching format:**
- NPZ files: `{'filenames': array([...]), 'embeddings': array(N, D)}`
- Separate cache file per embedding type
- Include metadata (model version, extraction date)

## UMAP Configuration

### Separate Projections

Compute 4 separate 2D UMAP projections:
1. DINOv2 embeddings only
2. CLIP embeddings only
3. PlantNet embeddings only
4. Combined (concatenated all three)

**UMAP Parameters:**
```python
umap.UMAP(
    n_neighbors=15,        # Balance local/global structure
    min_dist=0.1,          # Allow tight clusters
    metric='cosine',       # Best for high-dim embeddings
    n_components=2,        # 2D visualization
    random_state=42        # Reproducibility
)
```

### Why Separate UMAPs?

- Each embedding space has different geometry
- Compare how different models "see" botanical diversity
- Identify where models agree/disagree
- Combined UMAP shows consensus structure

**Performance:**
- ~30 seconds per UMAP on CPU
- Results cached for instant reloading
- Can recompute with different parameters without re-extracting embeddings

## Outlier Detection

### Multi-Method Approach

**1. Local Outlier Factor (LOF)**
- Applied to UMAP coordinates
- Detects points in sparse regions
- Catches badly segmented images (background, multiple plants)
- Sklearn implementation with n_neighbors=20

**2. Embedding Disagreement Score**
- Compare cluster assignments across 3 embeddings
- Use KMeans (k=30) on each embedding space
- Count how often an image changes clusters across embeddings
- High disagreement (3 different clusters) = potential outlier

**3. Distance from Nearest Neighbors**
- For each point, measure distance to 5th nearest neighbor
- Flag top 5% most distant points
- Catches unique specimens or artifacts

**4. Cluster Density Analysis**
- Identify very small clusters (< 3 images)
- Often indicates edge cases or rare specimens

### Confidence Scoring

```python
outliers = {
    'high_confidence': [...],      # Flagged by 3+ methods
    'medium_confidence': [...],    # Flagged by 2 methods
    'low_confidence': [...],       # Flagged by 1 method
    'method_details': {
        'image_path': ['LOF', 'disagreement', ...],
        ...
    }
}
```

### Outlier Report

HTML report including:
- Summary statistics (% flagged, method breakdown)
- Thumbnail grid organized by confidence level
- UMAP location highlighted for each outlier
- Detection reasons listed
- Links to full-resolution images
- Export button for flagged image list

## Interactive Visualization

### Dashboard Layout

**Tabbed Interface (Plotly Dash or standalone Plotly HTML):**
- Tab 1: DINOv2 UMAP
- Tab 2: CLIP UMAP
- Tab 3: PlantNet UMAP
- Tab 4: Combined UMAP
- Side panel: Controls and info

### Interactive Features

**1. Hover Tooltips:**
- 150x150px thumbnail image
- Filename
- Cluster ID
- Outlier score (if flagged)
- Nearest neighbor count

**2. Click Behavior:**
- Click point → opens full-resolution image in side panel
- Shows metadata:
  - Embedding distances to cluster center
  - Top 5 nearest neighbors with thumbnails
  - Outlier detection details
- "Show similar" button → highlights 10 nearest neighbors in plot

**3. Visualization Controls:**
- Toggle outliers:
  - Color-coded: red=high, orange=medium, yellow=low confidence
  - Show/hide each confidence level
- Color scheme selector:
  - Cluster coloring (discrete)
  - Density coloring (continuous)
  - Outlier highlighting
- Point size slider
- Opacity slider
- Download current view as PNG

**4. Cross-Embedding Navigation:**
- Click point in DINOv2 tab → automatically highlights same image in all tabs
- Shows how that plant appears across different embedding spaces
- Split-view mode: compare two embeddings side-by-side

**5. Selection and Export:**
- Lasso/box select regions
- Export selected points as:
  - CSV with coordinates and metadata
  - Image list (paths)
  - Thumbnail grid PNG
- Generate cluster report:
  - Images grouped by cluster
  - Cluster statistics
  - Representative images per cluster

### Technical Implementation

```python
class UMAPVisualizer:
    def create_interactive_dashboard(self, umap_results, embeddings, outliers):
        # Create Plotly figure with subplots for each embedding
        # Add scatter plots with custom hover data
        # Embed base64 thumbnails for offline viewing
        # Add interactive controls using Plotly callbacks
        # Generate standalone HTML (no server needed)
```

## Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing dependencies ...
    "umap-learn>=0.5.5",           # UMAP dimensionality reduction
    "transformers>=4.36.0",        # DINOv2, CLIP models
    "timm>=0.9.12",                # PlantNet models
    "plotly>=5.18.0",              # Interactive visualizations
    "scikit-learn>=1.3.0",         # Already present - LOF, KMeans
    "pillow>=10.0.0",              # Image loading (likely via opencv)
]
```

## Usage

### Command-Line Interface

```bash
# Full pipeline (extract, UMAP, visualize)
python plant_visualization.py --input masks/ --output visualizations/

# Skip extraction if cache exists
python plant_visualization.py --input masks/ --use-cache

# Recompute UMAP with different parameters
python plant_visualization.py --input masks/ --use-cache --recompute-umap \
    --n-neighbors 20 --min-dist 0.05

# Only generate outlier report
python plant_visualization.py --input masks/ --use-cache --outliers-only

# Clear cache and start fresh
python plant_visualization.py --input masks/ --clear-cache
```

### Python API

```python
from plant_visualization import EmbeddingExtractor, UMAPVisualizer, OutlierDetector

# Extract embeddings
extractor = EmbeddingExtractor(cache_dir='embeddings_cache/')
embeddings = extractor.extract_all(image_paths, use_cache=True)

# Compute UMAP
visualizer = UMAPVisualizer()
umap_results = visualizer.fit_transform(embeddings)

# Detect outliers
detector = OutlierDetector()
outliers = detector.detect(umap_results, embeddings)

# Create interactive dashboard
visualizer.create_dashboard(umap_results, embeddings, outliers,
                           output='visualizations/umap_interactive.html')
```

## Error Handling

- **GPU out of memory:** Auto-reduce batch size and retry
- **Model download failures:** Clear instructions for manual download
- **Corrupted images:** Skip with warning, log to error file
- **Interrupted processing:** Resume from last checkpoint
- **Cache version mismatch:** Detect and offer to recompute

## Performance Estimates

With 24GB GPU and 2,243 images:
- **Initial run (no cache):** 10-15 minutes
  - Model loading: 30 seconds
  - Embedding extraction: 8-12 minutes
  - UMAP computation: 2 minutes
  - Outlier detection: 30 seconds
  - Visualization generation: 1 minute

- **Cached run:** < 1 minute
  - Load cached embeddings: 5 seconds
  - Load cached UMAP: 2 seconds
  - Outlier detection: 30 seconds
  - Visualization generation: 1 minute

## Future Enhancements

Potential additions (not in initial implementation):
- Cluster labeling using CLIP text embeddings ("leaf shape", "flower", etc.)
- Time-series analysis if metadata includes collection dates
- Comparison with taxonomic labels from flora_meta.xlsx
- 3D UMAP option for deeper exploration
- Animation showing how clusters form with different n_neighbors
- Export to embedding search index for similarity queries

## Success Criteria

The implementation is successful if it:
1. Processes all 2,243 images without errors
2. Generates 4 interactive UMAP visualizations
3. Identifies outliers with reasonable precision (manual validation of top 50)
4. Dashboard is responsive and intuitive to use
5. Reveals interpretable botanical patterns in clusters
6. Cached reruns complete in < 1 minute
