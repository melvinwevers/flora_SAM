# Flora Batava Plant Explorer

An interactive Streamlit application for exploring 2,241 historical botanical illustrations from the Flora Batava collection.

## Features

- **Browse by Taxonomy**: Navigate through plant families, genera, and species
- **Visual Clusters**: Explore AI-discovered similarity groups using DINOv2 deep learning embeddings
- **Color Analysis**: Analyze color patterns and their relationships to taxonomy and clusters

## Quick Start

```bash
# From project root
uv sync

# Run the app
streamlit run streamlit_app/Home.py
```

The app will open in your default browser at `http://localhost:8501`.

## Directory Structure

```
streamlit_app/
├── Home.py                          # App entry point
├── pages/
│   ├── 1_Browse_by_Taxonomy.py     # Browse by plant families and genera
│   ├── 2_Visual_Clusters.py        # Explore AI-discovered clusters
│   ├── 3_Color_Analysis.py         # Color pattern analysis
│   └── Plant_Detail.py             # Individual plant details
├── data/
│   ├── flora_data.json             # Consolidated plant data (colors, taxonomy, clusters)
│   └── Flora Batava - illustration file names final.csv  # Source metadata
├── thumbnails/                      # 2,241 plant thumbnails (Git LFS)
└── utils/
    ├── data_loader.py              # Data loading with caching
    ├── color_utils.py              # Color utilities and categorization
    └── charts.py                   # Reusable Plotly components
```

## Data Files

All data files are in `streamlit_app/data/`:

- **flora_data.json** (12 MB): Consolidated plant data including:
  - 2,241 plants with colors (frequency, saliency, chroma rankings)
  - Full LAB color data for each color
  - Dimensions and mask percentages
  - Authoritative Flora Batava taxonomy and metadata
  - DINOv2 cluster assignments (23 visual similarity groups)
  - Precomputed taxonomy vs cluster comparison metrics
- **Flora Batava - illustration file names final.csv** (814 KB): Source illustration metadata (used during data generation)
- **thumbnails/** (~23 MB): 400px wide JPGs at quality 90, tracked with Git LFS

## Deploying to Streamlit Community Cloud

1. Push repository to GitHub
2. Ensure Git LFS is set up (see `SETUP_LFS.md` in streamlit_app directory)
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Connect your GitHub repository
5. Set main file path: `streamlit_app/Home.py`
6. Deploy!

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

## License

See project LICENSE file.
