# Color Feature Extraction Pipeline Design

**Date**: 2026-01-21
**Purpose**: Extract color features from segmented plant images and taxonomic data for analysis-ready dataset

## Overview

This pipeline implements Step 7 from [colorapproach.md](../../colorapproach.md) - extracting comprehensive color features from masked plant images and linking them to taxonomic data from Flora Batava Index.

The system loads color analysis results (LAB-based k-means clustering with saliency rankings), joins them with taxonomic metadata, computes per-image features, and calculates distinctiveness metrics relative to taxonomic groups.

## Architecture

The pipeline consists of three main layers:

### 1. Data Loading Layer
- **Color data**: Load `masks/masks_metadata.json` containing LAB color analysis
- **Taxonomic data**: Load `data/Flora Batava Index.xlsx` sheet "File names 1st each volume"
- **Joining strategy**: Strip `.jpg` from Excel "File no." column to match mask filenames

### 2. Feature Extraction Layer

**Per-image features** (computed from `colors_visual` - saliency-ranked colors):

**Basic LAB Statistics** (salience-weighted):
- Mean L*, a*, b* values weighted by visual_importance scores
- Standard deviation of L*, a*, b* values
- Median L*, a*, b* values (robust to outliers)

**Color Complexity Metrics**:
- **Color diversity**: Shannon entropy of visual_importance distribution
  - Formula: `-Σ(p_i * log(p_i))` where `p_i = visual_importance_i / Σ(visual_importance)`
- **Color spread**: Weighted standard deviation of LAB distances from mean
  - Uses visual_importance as weights for robust spread measure
- **Hue concentration**: Circular variance of hue angles (from LCH color space)
  - Low variance = concentrated hue range, high variance = diverse hues

**Color Category Features** (LAB/LCH thresholds):
- Red intensity: Proportion of saliency from red-range colors (hue 20-60°, chroma >20, L* >20)
- Yellow intensity: hue 60-100°, chroma >20, L* >30
- Green intensity: hue 100-180°, chroma >15, L* >20
- Blue intensity: hue 180-280°, chroma >15, L* >20
- Purple intensity: hue 280-340°, chroma >15, L* >20

**Conspicuousness**:
- Weighted mean Delta E distance from neutral gray [L*=50, a*=0, b*=0]
- Represents how attention-grabbing the color palette is

### 3. Distinctiveness Calculation Layer

After computing per-image features, calculate taxonomic group centroids:

**Global centroid**: Median LAB values across all images
**Family centroids**: Median LAB per botanical family
**Genus centroids**: Median LAB per genus (extracted from "Huidige botanische naam")

**Distinctiveness features** (LAB Delta E distances):
- `distinctiveness_global`: Distance from global centroid
- `distinctiveness_family`: Distance from own family's centroid
- `distinctiveness_genus`: Distance from own genus's centroid

## Feature Extraction Details

### Color Category Thresholds

Convert LAB to LCH (cylindrical representation):
- L* = lightness (unchanged)
- C* (chroma) = `sqrt(a*² + b*²)` - saturation magnitude
- H* (hue) = `atan2(b*, a*) * 180/π` - hue angle in degrees

**Red**: 20° ≤ H < 60°, C ≥ 20, L ≥ 20
**Yellow**: 60° ≤ H < 100°, C ≥ 20, L ≥ 30
**Green**: 100° ≤ H < 180°, C ≥ 15, L ≥ 20
**Blue**: 180° ≤ H < 280°, C ≥ 15, L ≥ 20
**Purple**: 280° ≤ H < 340°, C ≥ 15, L ≥ 20

Intensity = sum of visual_importance scores for colors matching category criteria.

### Circular Statistics for Hue

Hue angles are circular (0° = 360°), requiring circular statistics:

1. Convert hue angles to radians
2. Compute mean direction: `R = sqrt((Σcos(θ))² + (Σsin(θ))²) / n`
3. Circular variance: `1 - R` (ranges 0-1, lower = more concentrated)

Weight by visual_importance when computing mean direction.

### Salience Weighting

All statistical calculations use `visual_importance` as weights:

```python
# Weighted mean
weighted_mean = Σ(value_i * weight_i) / Σ(weight_i)

# Weighted standard deviation
weighted_var = Σ(weight_i * (value_i - weighted_mean)²) / Σ(weight_i)
weighted_std = sqrt(weighted_var)
```

This ensures visually prominent colors contribute more to aggregate statistics.

## Error Handling Strategy

**Missing taxonomic data**:
- If image filename not found in Excel: Fill taxonomic columns with `None`, compute features, skip distinctiveness
- If botanical family missing: Use genus-only grouping
- If genus extraction fails: Use family-only grouping

**Missing color data**:
- If `colors_visual` empty or missing: Fill all color features with `NaN`, log warning
- If some LAB values malformed: Skip those colors, continue with valid ones

**Robustness measures**:
- Use median for centroids (robust to outliers)
- Require minimum 3 images per group for centroid calculation (else mark distinctiveness as `NaN`)
- Log all data quality issues to `extraction_warnings.log`

**Graceful degradation**: Never crash the entire pipeline on single-image failures. Continue processing, mark problematic records, report at end.

## Implementation Structure

### Main Class: `ColorFeatureExtractor`

```python
class ColorFeatureExtractor:
    def __init__(self, masks_metadata_path, taxonomy_path):
        """Load data sources and prepare for extraction."""

    def extract_all_features(self):
        """Run complete pipeline, return DataFrame."""

    def _compute_per_image_features(self, colors_visual):
        """Extract features for single image."""

    def _compute_taxonomic_centroids(self, df):
        """Calculate group centroids from feature data."""

    def _compute_distinctiveness(self, df, centroids):
        """Add distinctiveness features to DataFrame."""

    def export_to_csv(self, df, output_path):
        """Save feature matrix to CSV."""
```

### Helper Functions

**LAB/LCH conversions**:
- `lab_to_lch(lab)` → (L, C, H)
- `lch_to_lab(lch)` → (L, a, b)

**Feature computations**:
- `compute_color_diversity(colors_visual)` → Shannon entropy
- `compute_color_spread(colors_visual, mean_lab)` → weighted std of distances
- `compute_hue_concentration(colors_visual)` → circular variance
- `compute_color_category_features(colors_visual)` → dict of intensities
- `compute_conspicuousness(colors_visual)` → mean Delta E from gray

**Distinctiveness**:
- `compute_distinctiveness_from_group(image_lab, group_centroid_lab)` → Delta E distance

### Data Flow

1. **Load**: Read masks_metadata.json and Excel taxonomy
2. **Join**: Match filenames (strip .jpg from File no.)
3. **Extract genus**: Split "Huidige botanische naam" on whitespace, take first word
4. **Per-image loop**:
   - Get `colors_visual` from metadata
   - Compute basic LAB stats (mean, std, median)
   - Compute complexity metrics (diversity, spread, hue concentration)
   - Compute color categories (red/yellow/green/blue/purple intensities)
   - Compute conspicuousness (distance from gray)
   - Store in row dict with taxonomic metadata
5. **Build DataFrame** from all rows
6. **Compute centroids**: Group by family/genus, calculate median LAB
7. **Compute distinctiveness**: For each image, calculate Delta E to relevant centroids
8. **Export CSVs**:
   - `color_features.csv`: Full feature matrix (one row per image)
   - `taxonomic_summary.csv`: Group statistics (one row per family/genus)

## Output Specifications

### color_features.csv

**Identifier columns**:
- `mask_file`: Mask filename (primary key)
- `image_file`: Original image filename
- `file_no`: Matched Excel File no. value
- `family`: Botanische familie
- `genus`: Extracted from Huidige botanische naam
- `species`: Full Huidige botanische naam

**Basic LAB statistics** (6 columns):
- `mean_L`, `mean_a`, `mean_b`: Salience-weighted means
- `std_L`, `std_a`, `std_b`: Salience-weighted standard deviations
- `median_L`, `median_a`, `median_b`: Robust central values

**Complexity metrics** (3 columns):
- `color_diversity`: Shannon entropy (higher = more diverse palette)
- `color_spread`: Weighted LAB distance spread (higher = more varied)
- `hue_concentration`: Circular variance (lower = concentrated hue range)

**Color categories** (5 columns):
- `red_intensity`, `yellow_intensity`, `green_intensity`, `blue_intensity`, `purple_intensity`

**Conspicuousness** (1 column):
- `conspicuousness`: Weighted Delta E from neutral gray

**Distinctiveness** (3 columns):
- `distinctiveness_global`: Delta E from overall centroid
- `distinctiveness_family`: Delta E from family centroid
- `distinctiveness_genus`: Delta E from genus centroid

**Total**: 28 columns

### taxonomic_summary.csv

**Group identifiers**:
- `family`: Botanische familie
- `genus`: Genus name (if applicable)

**Group statistics**:
- `n_images`: Number of images in group
- `centroid_L`, `centroid_a`, `centroid_b`: Median LAB values
- `mean_distinctiveness_global`: Average global distinctiveness
- `std_distinctiveness_global`: Variation in global distinctiveness

**Total**: 9 columns per row (family or genus level)

## Implementation Notes

1. **Use colors_visual exclusively**: This is saliency-ranked, already filtered for visual importance
2. **LAB space throughout**: All distance calculations use Delta E (Euclidean in LAB)
3. **Robust statistics**: Prefer median over mean for centroids (outlier resistance)
4. **Minimum group sizes**: Require 3+ images for centroid calculation
5. **Logging**: Track all data quality issues, missing values, and warnings
6. **Validation**: Check LAB ranges (L: 0-100, a/b: -128 to 127), log violations

## Testing Strategy

1. **Unit test helper functions**: LAB/LCH conversions, diversity/spread/concentration calculations
2. **Integration test on subset**: Process 10-20 images, verify feature ranges
3. **Validation checks**:
   - All features have expected ranges (no negative diversity, L* in 0-100)
   - Distinctiveness values are positive Delta E distances
   - No NaN values except for documented missing data cases
4. **Comparison with colorapproach.md**: Verify formulas match specification exactly

## Next Steps After Implementation

1. Run pipeline on full dataset (~8000 images)
2. Generate exploratory visualizations (distributions, correlations, PCA)
3. Export for downstream analysis (clustering, discriminant analysis, phylogenetic signal testing)
4. Document any edge cases or data quality issues discovered during full run
