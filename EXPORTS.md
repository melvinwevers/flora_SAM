# Color Data Exports

The `export_colors.py` script exports plant color data in analysis-friendly formats.

## Quick Start

```bash
# Export all formats to exports/ directory
python export_colors.py

# Export specific format only
python export_colors.py --format csv
python export_colors.py --format json
python export_colors.py --format summary
```

## Output Formats

### 1. Detailed CSV (`plant_colors_detailed.csv`)

**One row per color** - ideal for statistical analysis and data science.

```csv
plant_id,color_rank,hex,rgb_r,rgb_g,rgb_b,hsl_h,hsl_s,hsl_l,lab_l,lab_a,lab_b,frequency_pct,perceptual_weight,saliency_weight
001,1,#5a7a4c,90,122,76,108,23,38,47.2,-18.3,18.9,45.2,0.2156,0.1823
001,2,#3d4a2e,61,74,46,92,23,23,28.4,-12.1,13.2,22.8,0.1432,0.0945
...
```

**Use cases:**
- Statistical analysis in R/Python
- Color clustering and PCA
- Correlation with taxonomic features
- Machine learning training data

### 2. Nested JSON (`plant_colors.json`)

**One object per plant** with three color rankings.

```json
{
  "001": {
    "colors_frequency": [
      {
        "hex": "#5a7a4c",
        "rgb": [90, 122, 76],
        "hsl": [108, 23, 38],
        "lab": [47.2, -18.3, 18.9],
        "percentage": 45.2,
        "perceptual_weight": 0.2156,
        "saliency_weight": 0.1823
      },
      ...
    ],
    "colors_perceptual": [...],
    "colors_saliency": [...]
  }
}
```

**Use cases:**
- Web applications and APIs
- JavaScript/D3 visualizations
- Quick lookup by plant ID
- Integration with other tools

### 3. Summary CSV (`plant_colors_summary.csv`)

**One row per plant** with key color metrics.

```csv
plant_id,dominant_color_hex,dominant_color_pct,salient_color_hex,salient_color_pct,n_colors,color_diversity
001,#5a7a4c,45.2,#c23a42,12.3,5,1.4532
002,#6b8a5e,52.1,#e8d84f,8.7,5,1.3214
...
```

**Columns:**
- `dominant_color_hex`: Most frequent color
- `dominant_color_pct`: Coverage percentage
- `salient_color_hex`: Most attention-grabbing color
- `salient_color_pct`: Coverage percentage
- `n_colors`: Number of colors extracted
- `color_diversity`: Shannon entropy (higher = more diverse)

**Use cases:**
- Overview analysis
- Join with taxonomy data
- Quick comparisons between plants
- Identifying colorful vs. monochrome specimens

## Color Spaces Explained

**RGB** (0-255): Standard computer colors
- `rgb_r`, `rgb_g`, `rgb_b`

**HSL** (H: 0-360, S/L: 0-100): Perceptually meaningful
- `hsl_h`: Hue (color type: 0=red, 120=green, 240=blue)
- `hsl_s`: Saturation (vividness: 0=gray, 100=pure)
- `hsl_l`: Lightness (brightness: 0=black, 100=white)

**LAB** (L: 0-100, a/b: -128 to 127): Perceptually uniform
- `lab_l`: Lightness
- `lab_a`: Green (-) to Red (+)
- `lab_b`: Blue (-) to Yellow (+)
- Distance in LAB space = perceptual color difference

## Color Ranking Methods

**Frequency**: Ranked by area coverage (most common first)

**Perceptual**: Ranked by perceptual distinctiveness
- Formula: `0.4×frequency + 0.3×saturation + 0.3×contrast`
- Identifies colors that "pop" due to inherent properties

**Salience**: Ranked by spatial attention
- Uses spectral residual or edge-based saliency
- Identifies colors that naturally draw the eye
- Best for finding diagnostic botanical features

## Example Analysis Workflows

### Find Plants with Red Flowers

```python
import pandas as pd

# Load detailed colors
df = pd.DataFrame('exports/plant_colors_detailed.csv')

# Filter for red colors (hue 0-30 or 330-360) with high salience
red_salient = df[
    ((df['hsl_h'] < 30) | (df['hsl_h'] > 330)) &  # Red hue
    (df['hsl_s'] > 50) &  # Saturated
    (df['saliency_weight'] > 0.15)  # High salience
]

print(red_salient['plant_id'].unique())
```

### Color Diversity by Taxonomic Family

```python
import pandas as pd

summary = pd.read_csv('exports/plant_colors_summary.csv')
taxonomy = pd.read_csv('streamlit_app/data/plants_metadata.csv')

merged = summary.merge(taxonomy, on='plant_id')
diversity_by_family = merged.groupby('family')['color_diversity'].mean()

print(diversity_by_family.sort_values(ascending=False).head(10))
```

### Export for External Visualization

```bash
# Export to JSON for D3.js visualization
python export_colors.py --format json --output-dir public/data

# Use in web app
fetch('data/plant_colors.json')
  .then(r => r.json())
  .then(colors => {
    // Visualize with D3, Three.js, etc.
  });
```

## Options

```bash
python export_colors.py --help

Options:
  --metadata PATH        Input metadata JSON (default: masks/masks_metadata.json)
  --output-dir PATH      Output directory (default: exports)
  --format FORMAT        Export format: csv, json, summary, or all (default: all)
```

## Notes

- All exports are generated from `masks/masks_metadata.json`
- Color rankings require the new format with `perceptual_weight` and `saliency_weight` fields
- If old format detected, falls back to frequency ranking for all three
- Run `python color_analysis.py` first to ensure latest color data
