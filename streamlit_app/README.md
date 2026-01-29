# Flora Batava Plant Explorer - Streamlit App

An interactive Streamlit application for exploring 2,241 historical botanical illustrations from the Flora Batava collection.

## Features

- **Browse by Taxonomy**: Navigate through plant families, genera, and species
- **Visual Clusters**: Explore AI-discovered similarity groups using DINOv2, CLIP, PlantNet, and Combined embeddings
- **Color Analysis**: Analyze color patterns and their relationships to taxonomy and clusters

## Running Locally

### Prerequisites

- Python 3.8+
- Dependencies (install with `uv` or `pip`):
  - streamlit
  - pandas
  - plotly
  - Pillow

### Installation

```bash
# From project root
cd streamlit_app

# Install dependencies (if using uv from project root)
cd .. && uv sync

# Or install streamlit directly
pip install streamlit pandas plotly pillow
```

### Run the App

```bash
# From streamlit_app directory
streamlit run Home.py

# Or from project root
streamlit run streamlit_app/Home.py
```

The app will open in your default browser at `http://localhost:8501`.

## Directory Structure

```
streamlit_app/
├── app.py                 # Main entry point
├── Home.py                # Main entry point
├── pages/
│   ├── 1_Browse_by_Taxonomy.py
│   ├── 2_Visual_Clusters.py
│   └── 3_Color_Analysis
│   ├── plants_metadata.csv          # Plant metadata with colors
│   ├── cluster_assignments.csv      # Cluster memberships
│   └── comparison_metrics.json      # Precomputed analysis
├── thumbnails/                       # 2,241 plant thumbnails (23 MB, Git LFS)
└── utils/
    ├── data_loader.py               # Data loading with caching
    └── charts.py                    # Reusable Plotly components
```

## Data Files

- **plants_metadata.csv** (0.5 MB): 2,241 plants with colors, taxonomy, dimensions
- **cluster_assignments.csv** (0.07 MB): Cluster memberships for 4 models
- **comparison_metrics.json** (0.15 MB): Precomputed taxonomy vs cluster analysis
- **thumbnails/** (23 MB): 300px wide JPGs, tracked with Git LFS

## Deploying to Streamlit Community Cloud

1. Push this branch to GitHub
2. Ensure Git LFS is set up (see `SETUP_LFS.md`)
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Connect your GitHub repository
5. Set main file path: `streamlit_app/app.py`
6. Deploy!Home

Streamlit Cloud supports Git LFS out of the box.

## Development

### Adding New Features

- Data loading functions: Edit `utils/data_loader.py`
- Chart components: Edit `utils/charts.py`
- New pages: Add to `pages/` directory (numbered for order)

### Caching

All data loading functions use `@st.cache_data` for performance. Clear cache with:

```python
st.cache_data.clear()
```

Or use the "Clear cache" option in the Streamlit hamburger menu.

## Dataset Info

- **Source**: Flora Batava historical botanical illustrations
- **Plants**: 2,241 segmented specimens
- **Taxonomy**: 473 plants with family/genus data
- **Embeddings**: DINOv2 (23 clusters), CLIP (4 clusters), PlantNet (6 clusters), Combined (23 clusters)
- **Colors**: Top 5 by frequency and visual importance

## License

See project LICENSE file.
