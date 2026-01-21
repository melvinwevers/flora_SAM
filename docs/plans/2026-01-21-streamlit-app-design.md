# Streamlit Plant Explorer App Design

**Date:** 2026-01-21
**Status:** Approved
**Target:** Researchers/botanists exploring Flora Batava collection

## Overview

A Streamlit app hosted on GitHub (with LFS for thumbnails) that allows researchers to:
1. Browse plants by taxonomy (family/genus)
2. Explore AI-discovered visual clusters (DINOv2, CLIP, PlantNet)
3. Analyze color features and their relation to taxonomic/visual groups
4. Compare taxonomy vs. visual clusters to find patterns and outliers

## Architecture

### Deployment
- **Platform:** Streamlit Community Cloud
- **Repository:** GitHub with LFS for thumbnails
- **Future:** External image hosting (S3/GCS) for full-resolution images

### Directory Structure
```
streamlit_app/
├── app.py                 # Main entry point
├── pages/
│   ├── 1_Browse_by_Taxonomy.py
│   ├── 2_Visual_Clusters.py
│   ├── 3_Color_Analysis.py
│   └── 4_Plant_Detail.py
├── data/
│   ├── plants_metadata.csv       # Slimmed metadata
│   ├── color_features.csv        # From existing pipeline
│   ├── cluster_assignments.csv   # Plant → cluster mapping
│   ├── umap_coordinates.csv      # 2D coordinates per model
│   ├── comparison_metrics.json   # Precomputed analysis
│   └── cluster_stats.json        # Cluster sizes
├── thumbnails/                   # LFS-tracked, 300px JPGs
└── utils/
    ├── data_loader.py            # Cached data access
    └── charts.py                 # Reusable Plotly components
```

### Data Flow
```
Existing pipeline outputs
        ↓
Preprocessing script (generate_streamlit_data.py)
├── Resize segments → thumbnails/
├── Extract metadata → plants_metadata.csv
├── Extract UMAP coords → umap_coordinates.csv
├── Extract clusters → cluster_assignments.csv
└── Compute comparisons → comparison_metrics.json
        ↓
Streamlit app loads cached data
```

## Page Designs

### Page 1: Browse by Taxonomy

**Purpose:** Primary navigation through family/genus hierarchy

**Sidebar:**
- Family dropdown (57 families)
- Genus dropdown (filtered by selected family)
- Species search (optional text input)

**Main Area:**
- **Summary card:** N plants, N genera, dominant colors in group
- **Thumbnail grid:** Plants matching filters (paginated, ~20 per page)
- **Color distribution:** Bar chart of color categories in group
- **Cluster distribution:** Pie chart showing which visual clusters this taxonomic group falls into
- **Distinctiveness score:** How different this group's colors are from others

**Interactions:**
- Click thumbnail → navigate to Plant Detail page
- Click cluster in pie chart → navigate to Visual Clusters page filtered

---

### Page 2: Visual Clusters

**Purpose:** Explore AI-discovered similarity groups

**Sidebar:**
- Model selector: DINOv2 / CLIP / PlantNet / Combined
- Cluster selector: Dropdown with cluster sizes in parentheses
- Color-by toggle: Cluster ID vs. Family

**Main Area:**
- **UMAP scatter plot** (Plotly scattergl for performance)
  - 2,241 points
  - Hover shows thumbnail + metadata
  - Click selects cluster
  - Toggle coloring between cluster and family
- **Cluster summary panel:**
  - Size (N plants)
  - Top 3 families represented (with percentages)
  - Taxonomic purity score
  - Color profile (dominant colors in cluster)
- **Thumbnail grid:** Plants in selected cluster

**Key Insight:** When colored by family, visually see if taxonomic groups cluster together or scatter across visual space.

---

### Page 3: Color Analysis

**Purpose:** Analyze color patterns and taxonomy/cluster relationships

**Section A: Taxonomy vs. Clusters Comparison**
- **Contingency heatmap:** Rows = families (top 20 by size), Columns = clusters
  - Cell color = count of plants
  - Reveals which families dominate which clusters
- **Summary metrics:**
  - Adjusted Rand Index (overall agreement)
  - Per-cluster purity scores
  - Per-family concentration scores

**Section B: Color Feature Explorer**
- **Scatter plot:** Select X and Y axes from color features
  - Options: mean_L, mean_a, mean_b, color_diversity, conspicuousness, etc.
  - Color by: family, genus, or cluster
  - Hover shows plant thumbnail
- **Feature correlation heatmap:** Which color features correlate

**Section C: Surprising Plants**
- **Visually similar, taxonomically distant:**
  - Plants in same cluster but different families
  - Sorted by taxonomic distance
- **Taxonomically close, visually different:**
  - Plants in same genus but different clusters
  - Potential segmentation issues or true visual diversity
- Table with thumbnails, sortable and filterable

---

### Page 4: Plant Detail

**Purpose:** Deep dive into a single plant

**Layout:**
- **Header:** Plant filename, family, genus, species
- **Image:** Large thumbnail (300px), link to full-res (future)
- **Color palette:** 5 colors in both rankings (frequency + visual importance)
  - Swatches with hex codes and percentages
- **Color features table:** All extracted metrics
- **Similar plants grid:** Top 6 from same visual cluster
- **Related plants grid:** Top 6 from same genus
- **Cluster membership:** Which cluster in each model (DINOv2, CLIP, PlantNet)

---

## Comparison Methods

### 1. Contingency Heatmap
- Family × Cluster matrix
- Normalizable by row (family) or column (cluster)
- Shows taxonomic composition of clusters

### 2. UMAP with Taxonomy Coloring
- Same UMAP coordinates, different coloring
- Toggle reveals if families form visual clusters

### 3. Purity Scores
- **Cluster purity:** % of largest family in cluster
- **Family concentration:** % of family in largest cluster
- High values = alignment between visual and taxonomic grouping

### 4. Surprising Plants
- Cross-reference cluster and taxonomy
- Flag plants that break expected patterns
- Valuable for finding errors or interesting specimens

### 5. Statistical Agreement
- Adjusted Rand Index between cluster assignments and family labels
- Single metric summarizing overall correspondence

---

## Data Preparation

### Thumbnail Generation
```python
# generate_thumbnails.py
from PIL import Image
import os

THUMB_WIDTH = 300
INPUT_DIR = "masks/"
OUTPUT_DIR = "streamlit_app/thumbnails/"

for f in os.listdir(INPUT_DIR):
    if f.endswith("_segmented.png"):
        img = Image.open(f"{INPUT_DIR}/{f}")
        ratio = THUMB_WIDTH / img.width
        new_height = int(img.height * ratio)
        thumb = img.resize((THUMB_WIDTH, new_height), Image.LANCZOS)
        # Convert RGBA to RGB with white background for JPG
        if thumb.mode == 'RGBA':
            background = Image.new('RGB', thumb.size, (255, 255, 255))
            background.paste(thumb, mask=thumb.split()[3])
            thumb = background
        thumb_name = f.replace("_segmented.png", ".jpg")
        thumb.save(f"{OUTPUT_DIR}/{thumb_name}", "JPEG", quality=85)
```

### Metadata Extraction
Extract from `masks_metadata.json`:
- filename, family, genus, species
- colors_frequency, colors_visual (top 5 each)
- area_pixels, image_dimensions

Link with `color_features.csv` for analysis metrics.

### Cluster Data Extraction
From `cluster_data/*.json` and cached embeddings:
- Plant → cluster assignment per model
- UMAP x, y coordinates per model

### Comparison Metrics Precomputation
```python
# Contingency matrix
contingency = pd.crosstab(df['family'], df['cluster_dinov2'])

# Purity scores
cluster_purity = df.groupby('cluster_dinov2')['family'].apply(
    lambda x: x.value_counts().iloc[0] / len(x)
)

# Surprising plants
same_cluster_diff_family = df.groupby('cluster_dinov2').apply(
    lambda g: g[g['family'] != g['family'].mode()[0]]
)
```

---

## Git LFS Setup

```bash
cd streamlit_app
git lfs install
git lfs track "thumbnails/*.jpg"
git add .gitattributes
git commit -m "Track thumbnails with LFS"
```

**Estimated sizes:**
- Thumbnails: ~50-100MB (2,241 × 20-40KB)
- Data files: <10MB
- Total repo size: ~100-120MB

---

## Future Enhancements

1. **External image hosting:** Replace thumbnail paths with URLs (S3/GCS)
2. **Full-resolution viewer:** Modal or separate page with zoomable full image
3. **Export functionality:** Download filtered data as CSV
4. **Bookmark/favorites:** Save interesting plants for later
5. **Annotation system:** Flag segmentation errors or interesting specimens
6. **Additional embeddings:** Add more models for comparison

---

## Implementation Order

1. **Phase 1: Data preparation**
   - [ ] Generate thumbnails
   - [ ] Create slimmed metadata CSV
   - [ ] Extract UMAP coordinates and cluster assignments
   - [ ] Precompute comparison metrics
   - [ ] Set up Git LFS

2. **Phase 2: Core app**
   - [ ] App skeleton with navigation
   - [ ] Data loader with caching
   - [ ] Browse by Taxonomy page
   - [ ] Plant Detail page

3. **Phase 3: Cluster exploration**
   - [ ] Visual Clusters page
   - [ ] UMAP plot with toggle coloring

4. **Phase 4: Analysis features**
   - [ ] Color Analysis page
   - [ ] Contingency heatmap
   - [ ] Surprising plants table

5. **Phase 5: Deployment**
   - [ ] Deploy to Streamlit Community Cloud
   - [ ] Test with LFS
   - [ ] Documentation
