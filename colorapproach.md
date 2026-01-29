# Color Feature Extraction from Botanical Illustrations

## Project Overview

Extract and analyze color features from botanical illustrations to:
1. Understand color variation within taxonomic groups
2. Create color-based features for downstream analysis (e.g., survival models)
3. Characterize color distinctiveness and salience patterns

## Data Inputs

- Collection of plant illustrations (images)
- Extracted dominant and salient colors per image (RGB format with salience scores)
- Image clustering results
- Plant taxonomies (family, genus, species)

## Implementation Steps

### 1. Color Space Conversion

Convert all RGB color values to LAB color space for perceptually-uniform distance calculations.
```python
from skimage.color import rgb2lab, lab2lch
import numpy as np

def rgb_to_lab(rgb_array):
    """
    Convert RGB values to LAB color space.

    Parameters:
    -----------
    rgb_array : np.array, shape (n_colors, 3)
        RGB values in range [0, 255]

    Returns:
    --------
    lab_array : np.array, shape (n_colors, 3)
        LAB values: L* [0-100], a* [-128,127], b* [-128,127]
    """
    # Normalize RGB to [0, 1]
    rgb_normalized = rgb_array / 255.0
    # Reshape for skimage if needed
    rgb_reshaped = rgb_normalized.reshape(1, -1, 3)
    lab = rgb2lab(rgb_reshaped)
    return lab.reshape(-1, 3)

def lab_to_lch(lab_array):
    """
    Convert LAB to LCH (cylindrical representation).
    Useful for hue analysis.

    Returns:
    --------
    lch_array : np.array, shape (n_colors, 3)
        LCH values: L* [0-100], C* (chroma), H* (hue angle in degrees)
    """
    lch = lab2lch(lab_array.reshape(1, -1, 3))
    return lch.reshape(-1, 3)
```

**Rationale**: LAB space provides perceptually uniform distances. Equal Euclidean distances in LAB correspond to roughly equal perceived color differences, making it ideal for measuring color similarity and distinctiveness.

### 2. Basic Color Features (Per Image)

Extract fundamental color characteristics from each illustration.
```python
class ColorFeatureExtractor:
    def __init__(self, colors_rgb, salience_scores=None):
        """
        Parameters:
        -----------
        colors_rgb : np.array, shape (n_colors, 3)
            Dominant colors in RGB [0-255]
        salience_scores : np.array, shape (n_colors,), optional
            Salience score for each color
        """
        self.colors_rgb = colors_rgb
        self.colors_lab = rgb_to_lab(colors_rgb)
        self.colors_lch = lab_to_lch(self.colors_lab)
        self.salience = salience_scores if salience_scores is not None else np.ones(len(colors_rgb))
        self.salience = self.salience / self.salience.sum()  # Normalize to probabilities

    def get_color_count(self):
        """Number of distinct dominant colors."""
        return len(self.colors_rgb)

    def get_mean_color_lab(self, weighted=True):
        """
        Average color in LAB space.

        Parameters:
        -----------
        weighted : bool
            If True, weight by salience scores
        """
        if weighted:
            return np.average(self.colors_lab, axis=0, weights=self.salience)
        return np.mean(self.colors_lab, axis=0)

    def get_color_statistics(self):
        """
        Compute statistical summaries of colors.

        Returns:
        --------
        dict with keys:
            - mean_L, std_L: Lightness statistics
            - mean_a, std_a: a* (red-green) statistics
            - mean_b, std_b: b* (blue-yellow) statistics
            - mean_chroma, std_chroma: Chroma (saturation) statistics
            - mean_hue, std_hue: Hue angle statistics
        """
        L, a, b = self.colors_lab.T
        _, C, H = self.colors_lch.T

        return {
            'mean_L': np.average(L, weights=self.salience),
            'std_L': np.sqrt(np.average((L - np.average(L, weights=self.salience))**2, weights=self.salience)),
            'mean_a': np.average(a, weights=self.salience),
            'std_a': np.sqrt(np.average((a - np.average(a, weights=self.salience))**2, weights=self.salience)),
            'mean_b': np.average(b, weights=self.salience),
            'std_b': np.sqrt(np.average((b - np.average(b, weights=self.salience))**2, weights=self.salience)),
            'mean_chroma': np.average(C, weights=self.salience),
            'std_chroma': np.sqrt(np.average((C - np.average(C, weights=self.salience))**2, weights=self.salience)),
            'mean_hue': np.average(H, weights=self.salience),
            'std_hue': np.sqrt(np.average((H - np.average(H, weights=self.salience))**2, weights=self.salience)),
        }
```

### 3. Color Complexity and Diversity Features

Measure how varied and complex the color palette is for each illustration.
```python
from scipy.stats import entropy
from scipy.spatial.distance import pdist

def compute_color_diversity(colors_lab, salience):
    """
    Shannon entropy of color distribution.
    Higher values = more diverse color palette.

    Parameters:
    -----------
    colors_lab : np.array, shape (n_colors, 3)
    salience : np.array, shape (n_colors,)
        Normalized salience scores (sum to 1)
    """
    return entropy(salience)

def compute_color_spread(colors_lab, salience):
    """
    Salience-weighted average pairwise distance between colors.
    Measures how spread out colors are in LAB space.

    Returns:
    --------
    float : Average pairwise distance in LAB space
    """
    if len(colors_lab) < 2:
        return 0.0

    # Compute all pairwise distances
    pairwise_dists = pdist(colors_lab, metric='euclidean')

    # Weight by product of saliences
    n_colors = len(colors_lab)
    weights = []
    idx = 0
    for i in range(n_colors):
        for j in range(i + 1, n_colors):
            weights.append(salience[i] * salience[j])
            idx += 1

    weights = np.array(weights)
    weights = weights / weights.sum()

    return np.average(pairwise_dists, weights=weights)

def compute_hue_concentration(colors_lch, salience):
    """
    Measure concentration of hues.
    Low values = hues clustered (e.g., all greens)
    High values = hues spread across color wheel

    Uses circular variance for hue angles.
    """
    hues = colors_lch[:, 2]  # Hue angle in degrees

    # Convert to radians for circular statistics
    hues_rad = np.deg2rad(hues)

    # Circular mean
    sin_mean = np.average(np.sin(hues_rad), weights=salience)
    cos_mean = np.average(np.cos(hues_rad), weights=salience)

    # Circular variance (0 = concentrated, 1 = uniform)
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    circular_variance = 1 - R

    return circular_variance
```

### 4. Color Distinctiveness Features

Measure how unusual or distinctive each plant's colors are relative to reference groups.
```python
def compute_distinctiveness_from_group(plant_colors_lab, group_colors_lab,
                                       plant_salience=None, method='centroid'):
    """
    Compute how distinct a plant's colors are from its taxonomic group.

    Parameters:
    -----------
    plant_colors_lab : np.array, shape (n_plant_colors, 3)
        Colors of the focal plant
    group_colors_lab : list of np.arrays
        Colors from all plants in the taxonomic group
    plant_salience : np.array, shape (n_plant_colors,), optional
        Salience weights for plant colors
    method : str
        'centroid': Distance from group centroid
        'nearest': Distance to nearest group member
        'hausdorff': Maximum distance to nearest group member

    Returns:
    --------
    float : Distinctiveness score
    """
    # Flatten group colors to single array
    group_colors_flat = np.vstack(group_colors_lab)

    if method == 'centroid':
        # Compute group centroid (median is more robust than mean)
        group_centroid = np.median(group_colors_flat, axis=0)

        # Distance from plant's salient colors to centroid
        if plant_salience is not None:
            plant_salience = plant_salience / plant_salience.sum()
            distances = np.linalg.norm(plant_colors_lab - group_centroid, axis=1)
            return np.average(distances, weights=plant_salience)
        else:
            distances = np.linalg.norm(plant_colors_lab - group_centroid, axis=1)
            return np.mean(distances)

    elif method == 'nearest':
        # Distance to nearest color in group
        min_distances = []
        for plant_color in plant_colors_lab:
            dists = np.linalg.norm(group_colors_flat - plant_color, axis=1)
            min_distances.append(np.min(dists))

        if plant_salience is not None:
            plant_salience = plant_salience / plant_salience.sum()
            return np.average(min_distances, weights=plant_salience)
        else:
            return np.mean(min_distances)

    elif method == 'hausdorff':
        # Hausdorff distance (maximum of minimum distances)
        min_distances = []
        for plant_color in plant_colors_lab:
            dists = np.linalg.norm(group_colors_flat - plant_color, axis=1)
            min_distances.append(np.min(dists))
        return np.max(min_distances)

def compute_global_distinctiveness(plant_colors_lab, all_colors_lab, plant_salience=None):
    """
    Compute distinctiveness relative to entire collection.

    Returns percentile rank: 0 = most common colors, 100 = most unusual colors
    """
    dist = compute_distinctiveness_from_group(
        plant_colors_lab, all_colors_lab, plant_salience, method='centroid'
    )
    return dist
```

### 5. Specific Color Category Features

Extract features for botanically or aesthetically significant color categories.
```python
def compute_color_category_features(colors_lab, colors_lch, salience):
    """
    Compute intensity scores for specific color categories.

    Returns:
    --------
    dict with keys:
        - red_intensity: Strength of red colors
        - blue_intensity: Strength of blue colors
        - yellow_intensity: Strength of yellow colors
        - green_intensity: Strength of green colors
        - purple_intensity: Strength of purple colors
        - saturation_mean: Average chroma/saturation
        - brightness_mean: Average lightness
    """
    L, a, b = colors_lab.T
    _, C, H = colors_lch.T

    # Define color regions in LAB/LCH space
    # Red: positive a*, hue ~0-60° or 330-360°
    red_mask = (a > 20) & ((H < 60) | (H > 330))

    # Yellow: positive b*, hue ~60-120°
    yellow_mask = (b > 20) & (H >= 60) & (H < 120)

    # Green: negative a*, positive b*, hue ~120-180°
    green_mask = (a < -10) & (b > 0) & (H >= 120) & (H < 180)

    # Blue: negative b*, hue ~180-270°
    blue_mask = (b < -10) & (H >= 180) & (H < 270)

    # Purple: negative b*, positive a*, hue ~270-330°
    purple_mask = (a > 0) & (b < 0) & (H >= 270) & (H < 330)

    return {
        'red_intensity': np.sum(salience[red_mask]),
        'yellow_intensity': np.sum(salience[yellow_mask]),
        'green_intensity': np.sum(salience[green_mask]),
        'blue_intensity': np.sum(salience[blue_mask]),
        'purple_intensity': np.sum(salience[purple_mask]),
        'saturation_mean': np.average(C, weights=salience),
        'brightness_mean': np.average(L, weights=salience),
    }

def compute_conspicuousness(colors_lab, salience):
    """
    Compute how conspicuous/eye-catching colors are.

    High conspicuousness = highly saturated, distant from neutral gray.
    """
    # Neutral reference point in LAB
    neutral = np.array([50, 0, 0])  # Mid-gray

    # Distance from neutral for each color
    distances = np.linalg.norm(colors_lab - neutral, axis=1)

    # Weight by salience
    conspicuousness = np.average(distances, weights=salience)

    return conspicuousness
```

### 6. Taxonomic Group Analysis

Analyze color variation within and between taxonomic groups.
```python
from scipy.spatial import ConvexHull

class TaxonomicColorAnalyzer:
    def __init__(self, plants_df):
        """
        Parameters:
        -----------
        plants_df : pd.DataFrame with columns:
            - plant_id
            - colors_lab: list/array of LAB colors
            - salience: list/array of salience scores
            - family, genus, species: taxonomic information
        """
        self.df = plants_df

    def compute_within_group_dispersion(self, group_column='family'):
        """
        For each taxonomic group, compute color dispersion metrics.

        Returns:
        --------
        pd.DataFrame with columns:
            - [group_column]: taxonomic group name
            - n_plants: number of plants in group
            - mean_pairwise_distance: average LAB distance between all plant colors
            - std_pairwise_distance: standard deviation of distances
            - color_space_volume: volume of convex hull in LAB space (if ≥4 colors)
            - centroid_L, centroid_a, centroid_b: group centroid in LAB
        """
        results = []

        for group_name, group_df in self.df.groupby(group_column):
            # Collect all colors from all plants in group
            all_colors = []
            for colors in group_df['colors_lab']:
                all_colors.append(colors)
            all_colors = np.vstack(all_colors)

            # Pairwise distances
            if len(all_colors) > 1:
                pairwise_dists = pdist(all_colors, metric='euclidean')
                mean_dist = np.mean(pairwise_dists)
                std_dist = np.std(pairwise_dists)
            else:
                mean_dist = 0
                std_dist = 0

            # Convex hull volume (requires at least 4 points in 3D)
            volume = np.nan
            if len(all_colors) >= 4:
                try:
                    hull = ConvexHull(all_colors)
                    volume = hull.volume
                except:
                    pass  # Degenerate case (e.g., all colors coplanar)

            # Centroid
            centroid = np.median(all_colors, axis=0)

            results.append({
                group_column: group_name,
                'n_plants': len(group_df),
                'n_colors': len(all_colors),
                'mean_pairwise_distance': mean_dist,
                'std_pairwise_distance': std_dist,
                'color_space_volume': volume,
                'centroid_L': centroid[0],
                'centroid_a': centroid[1],
                'centroid_b': centroid[2],
            })

        return pd.DataFrame(results)

    def compute_between_group_distances(self, group_column='family'):
        """
        Compute distances between taxonomic group centroids.

        Returns:
        --------
        pd.DataFrame: pairwise distance matrix between groups
        """
        # Get group centroids
        centroids = {}
        for group_name, group_df in self.df.groupby(group_column):
            all_colors = np.vstack(group_df['colors_lab'].tolist())
            centroids[group_name] = np.median(all_colors, axis=0)

        # Compute pairwise distances
        group_names = list(centroids.keys())
        n_groups = len(group_names)
        dist_matrix = np.zeros((n_groups, n_groups))

        for i, name_i in enumerate(group_names):
            for j, name_j in enumerate(group_names):
                if i != j:
                    dist = np.linalg.norm(centroids[name_i] - centroids[name_j])
                    dist_matrix[i, j] = dist

        return pd.DataFrame(dist_matrix, index=group_names, columns=group_names)

    def identify_color_overlap(self, group_column='family', threshold=20):
        """
        Identify taxonomic groups with overlapping color spaces.

        Parameters:
        -----------
        threshold : float
            LAB distance threshold for considering colors "overlapping"

        Returns:
        --------
        list of tuples: (group1, group2, overlap_score)
        """
        between_dists = self.compute_between_group_distances(group_column)

        overlaps = []
        for i, group_i in enumerate(between_dists.index):
            for j, group_j in enumerate(between_dists.columns):
                if i < j:  # Upper triangle only
                    dist = between_dists.iloc[i, j]
                    if dist < threshold:
                        overlaps.append((group_i, group_j, dist))

        return sorted(overlaps, key=lambda x: x[2])
```

### 7. Complete Pipeline

Bringing it all together into a processing pipeline.
```python
import pandas as pd

def process_plant_collection(plants_df):
    """
    Extract all color features for a collection of plants.

    Parameters:
    -----------
    plants_df : pd.DataFrame with columns:
        - plant_id
        - colors_rgb: array of RGB colors shape (n_colors, 3)
        - salience: array of salience scores shape (n_colors,)
        - family, genus, species: taxonomic info

    Returns:
    --------
    features_df : pd.DataFrame with all extracted color features
    """
    features = []

    # Convert all colors to LAB
    plants_df['colors_lab'] = plants_df['colors_rgb'].apply(rgb_to_lab)
    plants_df['colors_lch'] = plants_df['colors_lab'].apply(
        lambda x: lab_to_lch(x)
    )

    # Collect all colors for global statistics
    all_colors_lab = np.vstack(plants_df['colors_lab'].tolist())

    # Process each plant
    for idx, row in plants_df.iterrows():
        extractor = ColorFeatureExtractor(
            row['colors_rgb'],
            row['salience']
        )

        # Basic features
        features_dict = {
            'plant_id': row['plant_id'],
            'color_count': extractor.get_color_count(),
        }

        # Statistical features
        features_dict.update(extractor.get_color_statistics())

        # Complexity features
        features_dict['color_diversity'] = compute_color_diversity(
            extractor.colors_lab, extractor.salience
        )
        features_dict['color_spread'] = compute_color_spread(
            extractor.colors_lab, extractor.salience
        )
        features_dict['hue_concentration'] = compute_hue_concentration(
            extractor.colors_lch, extractor.salience
        )

        # Color category features
        features_dict.update(compute_color_category_features(
            extractor.colors_lab, extractor.colors_lch, extractor.salience
        ))

        # Conspicuousness
        features_dict['conspicuousness'] = compute_conspicuousness(
            extractor.colors_lab, extractor.salience
        )

        # Distinctiveness - global
        features_dict['distinctiveness_global'] = compute_global_distinctiveness(
            extractor.colors_lab, [all_colors_lab], extractor.salience
        )

        # Distinctiveness - within family
        family_colors = plants_df[
            plants_df['family'] == row['family']
        ]['colors_lab'].tolist()
        features_dict['distinctiveness_family'] = compute_distinctiveness_from_group(
            extractor.colors_lab, family_colors, extractor.salience
        )

        # Distinctiveness - within genus
        genus_colors = plants_df[
            plants_df['genus'] == row['genus']
        ]['colors_lab'].tolist()
        features_dict['distinctiveness_genus'] = compute_distinctiveness_from_group(
            extractor.colors_lab, genus_colors, extractor.salience
        )

        features.append(features_dict)

    return pd.DataFrame(features)

def analyze_taxonomic_colors(plants_df):
    """
    Perform taxonomic-level color analysis.

    Returns:
    --------
    dict with keys:
        - family_dispersion: DataFrame of within-family color metrics
        - genus_dispersion: DataFrame of within-genus color metrics
        - family_distances: DataFrame of between-family distances
        - family_overlaps: List of overlapping families
    """
    analyzer = TaxonomicColorAnalyzer(plants_df)

    return {
        'family_dispersion': analyzer.compute_within_group_dispersion('family'),
        'genus_dispersion': analyzer.compute_within_group_dispersion('genus'),
        'family_distances': analyzer.compute_between_group_distances('family'),
        'family_overlaps': analyzer.identify_color_overlap('family'),
    }
```

### 8. Output Data Structure

The pipeline produces two main outputs:

**Color Features Dataset** (`color_features.csv`):
```
plant_id, color_count, mean_L, std_L, mean_a, std_a, mean_b, std_b,
mean_chroma, std_chroma, mean_hue, std_hue, color_diversity, color_spread,
hue_concentration, red_intensity, yellow_intensity, green_intensity,
blue_intensity, purple_intensity, saturation_mean, brightness_mean,
conspicuousness, distinctiveness_global, distinctiveness_family,
distinctiveness_genus
```

**Taxonomic Color Summary** (`taxonomic_color_analysis.csv`):
```
family/genus, n_plants, n_colors, mean_pairwise_distance,
std_pairwise_distance, color_space_volume, centroid_L, centroid_a, centroid_b
```

### 9. Visualization Functions
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def plot_lab_space_3d(plants_df, color_by='family', alpha=0.6):
    """
    3D scatter plot of colors in LAB space.

    Parameters:
    -----------
    plants_df : DataFrame with colors_lab column
    color_by : str
        Column name to color points by (e.g., 'family', 'genus')
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Flatten all colors
    plot_data = []
    for idx, row in plants_df.iterrows():
        for color_lab in row['colors_lab']:
            plot_data.append({
                'L': color_lab[0],
                'a': color_lab[1],
                'b': color_lab[2],
                color_by: row[color_by]
            })

    plot_df = pd.DataFrame(plot_data)

    # Color by category
    categories = plot_df[color_by].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))

    for i, category in enumerate(categories):
        subset = plot_df[plot_df[color_by] == category]
        ax.scatter(subset['L'], subset['a'], subset['b'],
                  c=[colors[i]], label=category, alpha=alpha, s=20)

    ax.set_xlabel('L* (Lightness)')
    ax.set_ylabel('a* (Red-Green)')
    ax.set_zlabel('b* (Blue-Yellow)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'Color Distribution in LAB Space (colored by {color_by})')

    return fig

def plot_taxonomic_dispersion(dispersion_df, group_column='family', top_n=20):
    """
    Bar plot of color dispersion by taxonomic group.
    """
    # Sort by mean pairwise distance
    dispersion_sorted = dispersion_df.nlargest(top_n, 'mean_pairwise_distance')

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(dispersion_sorted))
    ax.barh(y_pos, dispersion_sorted['mean_pairwise_distance'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dispersion_sorted[group_column])
    ax.set_xlabel('Mean Pairwise Color Distance (LAB)')
    ax.set_title(f'Color Dispersion by {group_column.capitalize()} (Top {top_n})')

    plt.tight_layout()
    return fig

def plot_color_features_correlation(features_df):
    """
    Correlation heatmap of color features.
    """
    # Select only numeric feature columns
    feature_cols = [col for col in features_df.columns
                   if col not in ['plant_id'] and features_df[col].dtype in ['float64', 'int64']]

    corr = features_df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Color Feature Correlations')
    plt.tight_layout()

    return fig
```

### 10. Required Python Libraries
```python
# Core
import numpy as np
import pandas as pd

# Color processing
from skimage.color import rgb2lab, lab2lch

# Statistics and distances
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
```

### 11. Usage Example
```python
# Load your data
plants_df = pd.read_csv('plants_data.csv')

# Assuming you have:
# - colors_rgb: list of RGB arrays per plant
# - salience: list of salience scores per plant
# - family, genus, species: taxonomic columns

# Extract all color features
color_features = process_plant_collection(plants_df)
color_features.to_csv('color_features.csv', index=False)

# Analyze taxonomic patterns
taxonomic_analysis = analyze_taxonomic_colors(plants_df)
taxonomic_analysis['family_dispersion'].to_csv('family_color_dispersion.csv', index=False)
taxonomic_analysis['genus_dispersion'].to_csv('genus_color_dispersion.csv', index=False)

# Generate visualizations
fig1 = plot_lab_space_3d(plants_df, color_by='family')
fig1.savefig('lab_space_by_family.png', dpi=300, bbox_inches='tight')

fig2 = plot_taxonomic_dispersion(taxonomic_analysis['family_dispersion'])
fig2.savefig('family_dispersion.png', dpi=300, bbox_inches='tight')

fig3 = plot_color_features_correlation(color_features)
fig3.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
```

## Notes on Color Binning

You asked about binning colors into RGB or LAB cubes. Based on your use case, here's guidance:

**Skip binning if**: You have manageable data size and want maximum information preservation. Use raw LAB values as continuous features.

**Use binning if**:
- You need to create discrete color "vocabularies" across your collection
- You're dealing with very large color sets and need dimensionality reduction
- You want to compare color usage across groups at a coarser level

**If binning, use LAB not RGB**:
- LAB bins create perceptually uniform buckets
- Bin edges: L* in steps of 10 (0-100), a* and b* in steps of 20 (-128 to 127)
- This creates roughly 10 × 13 × 13 ≈ 1,690 bins
- Adjust granularity based on your collection's color distribution

However, for your stated goals (understanding within-taxonomy variation + creating features for downstream models), continuous LAB features are likely more appropriate than binning.
