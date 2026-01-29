#!/usr/bin/env python3
"""
Color Feature Extraction Pipeline

Extracts comprehensive color features from segmented plant images and links them
to taxonomic data for analysis-ready dataset export.

Features extracted:
- Basic LAB statistics (mean, std, median)
- Color complexity metrics (diversity, spread, hue concentration)
- Color category intensities (red/yellow/green/blue/purple)
- Conspicuousness (distance from neutral gray)
- Distinctiveness (relative to global/family/genus centroids)
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from skimage.color import rgb2lab

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction_warnings.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def lab_to_lch(lab):
    """
    Convert LAB to LCH (cylindrical coordinates).

    Args:
        lab: Tuple or array of (L, a, b) values

    Returns:
        Tuple of (L, C, H) where:
        - L: Lightness (0-100)
        - C: Chroma/saturation magnitude (0+)
        - H: Hue angle in degrees (0-360)
    """
    L, a, b = lab
    C = np.sqrt(a**2 + b**2)  # Chroma
    H = np.arctan2(b, a) * 180 / np.pi  # Hue in degrees
    if H < 0:
        H += 360  # Normalize to [0, 360]
    return (L, C, H)


def lch_to_lab(lch):
    """
    Convert LCH to LAB (Cartesian coordinates).

    Args:
        lch: Tuple of (L, C, H) values

    Returns:
        Tuple of (L, a, b)
    """
    L, C, H = lch
    H_rad = H * np.pi / 180  # Convert to radians
    a = C * np.cos(H_rad)
    b = C * np.sin(H_rad)
    return (L, a, b)


def color_distance_lab(lab1, lab2):
    """
    Calculate Delta E distance between two LAB colors.

    Args:
        lab1: Tuple or array of (L, a, b)
        lab2: Tuple or array of (L, a, b)

    Returns:
        Delta E distance
    """
    lab1 = np.array(lab1)
    lab2 = np.array(lab2)
    return np.sqrt(np.sum((lab1 - lab2)**2))


def compute_color_diversity(colors_visual):
    """
    Compute Shannon entropy of visual importance distribution.
    Higher entropy = more diverse color palette.

    Args:
        colors_visual: List of color dicts with 'visual_importance' field

    Returns:
        Shannon entropy value (0+ , higher = more diverse)
    """
    if not colors_visual or len(colors_visual) == 0:
        return np.nan

    # Extract visual importance scores
    importances = np.array([c['visual_importance'] for c in colors_visual])

    # Normalize to probability distribution
    if importances.sum() == 0:
        return np.nan

    probabilities = importances / importances.sum()

    # Calculate Shannon entropy: -Σ(p * log(p))
    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log(probabilities))

    return entropy


def compute_color_spread(colors_visual, mean_lab):
    """
    Compute salience-weighted standard deviation of LAB distances from mean.
    Measures how spread out the colors are in perceptual space.

    Args:
        colors_visual: List of color dicts with 'lab' and 'visual_importance'
        mean_lab: Mean LAB color (L, a, b)

    Returns:
        Weighted standard deviation of distances
    """
    if not colors_visual or len(colors_visual) == 0:
        return np.nan

    # Extract LAB values and weights
    lab_values = np.array([c['lab'] for c in colors_visual])
    weights = np.array([c['visual_importance'] for c in colors_visual])

    if weights.sum() == 0:
        return np.nan

    # Calculate distances from mean
    distances = np.array([color_distance_lab(lab, mean_lab) for lab in lab_values])

    # Weighted variance
    weighted_mean_distance = np.average(distances, weights=weights)
    weighted_variance = np.average((distances - weighted_mean_distance)**2, weights=weights)

    return np.sqrt(weighted_variance)


def compute_hue_concentration(colors_visual):
    """
    Compute circular variance of hue angles.
    Lower variance = more concentrated hue range, higher = diverse hues.

    Uses circular statistics to handle hue angle periodicity (0° = 360°).

    Args:
        colors_visual: List of color dicts with 'lab' and 'visual_importance'

    Returns:
        Circular variance (0-1, lower = more concentrated)
    """
    if not colors_visual or len(colors_visual) == 0:
        return np.nan

    # Convert LAB to LCH to get hue angles
    hues = []
    weights = []
    for color in colors_visual:
        L, C, H = lab_to_lch(color['lab'])
        # Only include colors with sufficient chroma (avoid noise in hue for gray colors)
        if C > 5:  # Minimum chroma threshold
            hues.append(H)
            weights.append(color['visual_importance'])

    if not hues or len(hues) == 0:
        return np.nan

    # Convert to radians
    hues_rad = np.array(hues) * np.pi / 180
    weights = np.array(weights)

    if weights.sum() == 0:
        return np.nan

    # Weighted circular mean direction
    cos_sum = np.sum(weights * np.cos(hues_rad))
    sin_sum = np.sum(weights * np.sin(hues_rad))
    weight_sum = weights.sum()

    # Mean resultant length R
    R = np.sqrt(cos_sum**2 + sin_sum**2) / weight_sum

    # Circular variance = 1 - R
    return 1 - R


def compute_color_category_features(colors_visual):
    """
    Compute intensity scores for color categories based on LAB/LCH thresholds.

    Categories:
    - Red: 20° ≤ H < 60°, C ≥ 20, L ≥ 20
    - Yellow: 60° ≤ H < 100°, C ≥ 20, L ≥ 30
    - Green: 100° ≤ H < 180°, C ≥ 15, L ≥ 20
    - Blue: 180° ≤ H < 280°, C ≥ 15, L ≥ 20
    - Purple: 280° ≤ H < 340°, C ≥ 15, L ≥ 20

    Args:
        colors_visual: List of color dicts with 'lab' and 'visual_importance'

    Returns:
        Dict with category intensity scores (sum of visual importance for matching colors)
    """
    if not colors_visual or len(colors_visual) == 0:
        return {
            'red_intensity': np.nan,
            'yellow_intensity': np.nan,
            'green_intensity': np.nan,
            'blue_intensity': np.nan,
            'purple_intensity': np.nan
        }

    # Initialize category scores
    categories = {
        'red_intensity': 0.0,
        'yellow_intensity': 0.0,
        'green_intensity': 0.0,
        'blue_intensity': 0.0,
        'purple_intensity': 0.0
    }

    for color in colors_visual:
        L, C, H = lab_to_lch(color['lab'])
        importance = color['visual_importance']

        # Red: 20° ≤ H < 60°, C ≥ 20, L ≥ 20
        if 20 <= H < 60 and C >= 20 and L >= 20:
            categories['red_intensity'] += importance

        # Yellow: 60° ≤ H < 100°, C ≥ 20, L ≥ 30
        elif 60 <= H < 100 and C >= 20 and L >= 30:
            categories['yellow_intensity'] += importance

        # Green: 100° ≤ H < 180°, C ≥ 15, L ≥ 20
        elif 100 <= H < 180 and C >= 15 and L >= 20:
            categories['green_intensity'] += importance

        # Blue: 180° ≤ H < 280°, C ≥ 15, L ≥ 20
        elif 180 <= H < 280 and C >= 15 and L >= 20:
            categories['blue_intensity'] += importance

        # Purple: 280° ≤ H < 340°, C ≥ 15, L ≥ 20
        elif 280 <= H < 340 and C >= 15 and L >= 20:
            categories['purple_intensity'] += importance

    return categories


def compute_conspicuousness(colors_visual):
    """
    Compute weighted mean Delta E distance from neutral gray [L=50, a=0, b=0].
    Higher values = more attention-grabbing colors.

    Args:
        colors_visual: List of color dicts with 'lab' and 'visual_importance'

    Returns:
        Weighted mean distance from neutral gray
    """
    if not colors_visual or len(colors_visual) == 0:
        return np.nan

    neutral_gray = np.array([50, 0, 0])  # L=50, a=0, b=0

    # Calculate distances
    lab_values = np.array([c['lab'] for c in colors_visual])
    weights = np.array([c['visual_importance'] for c in colors_visual])

    if weights.sum() == 0:
        return np.nan

    distances = np.array([color_distance_lab(lab, neutral_gray) for lab in lab_values])

    # Weighted mean
    weighted_mean = np.average(distances, weights=weights)

    return weighted_mean


def compute_distinctiveness_from_group(image_lab, group_centroid_lab):
    """
    Compute Delta E distance from image's mean LAB to group centroid.

    Args:
        image_lab: Tuple of (L, a, b) for image mean color
        group_centroid_lab: Tuple of (L, a, b) for group centroid

    Returns:
        Delta E distance
    """
    if group_centroid_lab is None or any(np.isnan(group_centroid_lab)):
        return np.nan

    return color_distance_lab(image_lab, group_centroid_lab)


class ColorFeatureExtractor:
    """
    Extract comprehensive color features from segmented plant images.
    Links color data to taxonomic metadata and computes distinctiveness metrics.
    """

    def __init__(self, masks_metadata_path, taxonomy_path, meta_path='flora_meta.xlsx', taxonomy_sheet=0):
        """
        Initialize the feature extractor.

        Args:
            masks_metadata_path: Path to masks_metadata.json
            taxonomy_path: Path to Flora Batava Index.xlsx
            meta_path: Path to flora_meta.xlsx (filename to illustration order mapping)
            taxonomy_sheet: Sheet index or name (default: 0 = first sheet)
        """
        self.masks_metadata_path = Path(masks_metadata_path)
        self.taxonomy_path = Path(taxonomy_path)
        self.meta_path = Path(meta_path)
        self.taxonomy_sheet = taxonomy_sheet

        # Load data
        logger.info("Loading masks metadata...")
        with open(self.masks_metadata_path, 'r') as f:
            self.masks_metadata = json.load(f)
        logger.info(f"Loaded {len(self.masks_metadata)} mask records")

        # Load flora_meta.xlsx to get filename to illustration order mapping
        logger.info(f"Loading flora_meta from {meta_path}...")
        meta_df = pd.read_excel(meta_path)

        # Filter to illustrations only and add order
        illustrations = meta_df[meta_df['File type'] == 'illustration'].copy()
        illustrations = illustrations.reset_index(drop=True)
        illustrations['illustration_order'] = illustrations.index + 1  # 1-indexed

        # Create filename lookup (strip .jpg to match mask filenames)
        illustrations['file_key'] = illustrations['File name'].str.replace('.jpg', '', regex=False)
        self.illustrations_df = illustrations
        logger.info(f"Loaded {len(self.illustrations_df)} illustration records")

        # Load Flora Batava Index
        logger.info(f"Loading taxonomy data from {taxonomy_path}, sheet {taxonomy_sheet}...")
        self.taxonomy_df = pd.read_excel(taxonomy_path, sheet_name=taxonomy_sheet)
        logger.info(f"Loaded {len(self.taxonomy_df)} taxonomic records")

        # Create illustration order column for joining (1-indexed, matches row number)
        self.taxonomy_df['illustration_order'] = self.taxonomy_df.index + 1

        # Load sheet 3 to get family information (only 98 records but has family data)
        logger.info("Loading family data from sheet 'File names 1st each volume'...")
        family_sheet = pd.read_excel(taxonomy_path, sheet_name='File names 1st each volume')

        # Create both species → family and genus → family mappings
        family_data = family_sheet[['Huidige botanische naam', 'Botanische familie']].dropna()

        # Species → family mapping (exact match)
        self.species_family_map = (
            family_data
            .drop_duplicates('Huidige botanische naam')
            .set_index('Huidige botanische naam')['Botanische familie']
            .to_dict()
        )

        # Genus → family mapping (extracted from species names)
        family_data['genus'] = family_data['Huidige botanische naam'].apply(
            lambda x: x.split()[0] if pd.notna(x) and len(str(x).split()) > 0 else None
        )
        self.genus_family_map = (
            family_data[['genus', 'Botanische familie']]
            .dropna()
            .drop_duplicates('genus')
            .set_index('genus')['Botanische familie']
            .to_dict()
        )

        logger.info(
            f"Loaded family mappings: {len(self.species_family_map)} species, "
            f"{len(self.genus_family_map)} genera"
        )

    def _extract_genus(self, botanical_name):
        """
        Extract genus from botanical name (first word).

        Args:
            botanical_name: Full botanical name (e.g., "Quercus robur L.")

        Returns:
            Genus name (e.g., "Quercus") or None if extraction fails
        """
        if pd.isna(botanical_name) or botanical_name == '':
            return None

        parts = str(botanical_name).strip().split()
        if len(parts) > 0:
            return parts[0]
        return None

    def _compute_per_image_features(self, colors_visual):
        """
        Extract all features for a single image.

        Args:
            colors_visual: List of saliency-ranked color dicts

        Returns:
            Dict of feature values
        """
        features = {}

        if not colors_visual or len(colors_visual) == 0:
            # Return NaN for all features if no colors
            logger.warning("No colors_visual data, filling with NaN")
            return {
                'mean_L': np.nan, 'mean_a': np.nan, 'mean_b': np.nan,
                'std_L': np.nan, 'std_a': np.nan, 'std_b': np.nan,
                'median_L': np.nan, 'median_a': np.nan, 'median_b': np.nan,
                'color_diversity': np.nan,
                'color_spread': np.nan,
                'hue_concentration': np.nan,
                'red_intensity': np.nan,
                'yellow_intensity': np.nan,
                'green_intensity': np.nan,
                'blue_intensity': np.nan,
                'purple_intensity': np.nan,
                'conspicuousness': np.nan
            }

        # Extract LAB values and weights
        lab_values = np.array([c['lab'] for c in colors_visual])
        weights = np.array([c['visual_importance'] for c in colors_visual])

        # Check for valid weights
        if weights.sum() == 0:
            logger.warning("All visual_importance values are 0, using equal weights")
            weights = np.ones(len(weights))

        # Basic LAB statistics (salience-weighted)
        features['mean_L'] = np.average(lab_values[:, 0], weights=weights)
        features['mean_a'] = np.average(lab_values[:, 1], weights=weights)
        features['mean_b'] = np.average(lab_values[:, 2], weights=weights)

        # Weighted standard deviations
        features['std_L'] = np.sqrt(np.average((lab_values[:, 0] - features['mean_L'])**2, weights=weights))
        features['std_a'] = np.sqrt(np.average((lab_values[:, 1] - features['mean_a'])**2, weights=weights))
        features['std_b'] = np.sqrt(np.average((lab_values[:, 2] - features['mean_b'])**2, weights=weights))

        # Median values (robust to outliers)
        features['median_L'] = np.median(lab_values[:, 0])
        features['median_a'] = np.median(lab_values[:, 1])
        features['median_b'] = np.median(lab_values[:, 2])

        # Mean LAB for other calculations
        mean_lab = np.array([features['mean_L'], features['mean_a'], features['mean_b']])

        # Complexity metrics
        features['color_diversity'] = compute_color_diversity(colors_visual)
        features['color_spread'] = compute_color_spread(colors_visual, mean_lab)
        features['hue_concentration'] = compute_hue_concentration(colors_visual)

        # Color categories
        category_features = compute_color_category_features(colors_visual)
        features.update(category_features)

        # Conspicuousness
        features['conspicuousness'] = compute_conspicuousness(colors_visual)

        return features

    def _compute_taxonomic_centroids(self, df):
        """
        Calculate median LAB centroids for global, family, and genus groups.
        Only computes centroids for groups with 3+ images.

        Args:
            df: DataFrame with per-image features

        Returns:
            Dict with 'global', 'family', and 'genus' centroid mappings
        """
        centroids = {
            'global': None,
            'family': {},
            'genus': {}
        }

        # Global centroid (all images)
        if len(df) >= 3:
            centroids['global'] = (
                df['median_L'].median(),
                df['median_a'].median(),
                df['median_b'].median()
            )
            logger.info(f"Global centroid: L={centroids['global'][0]:.1f}, a={centroids['global'][1]:.1f}, b={centroids['global'][2]:.1f}")
        else:
            logger.warning("Not enough images for global centroid (need 3+)")

        # Family centroids
        family_groups = df.groupby('family')
        for family, group in family_groups:
            if pd.isna(family) or family == '' or len(group) < 3:
                continue
            centroids['family'][family] = (
                group['median_L'].median(),
                group['median_a'].median(),
                group['median_b'].median()
            )
        logger.info(f"Computed {len(centroids['family'])} family centroids")

        # Genus centroids
        genus_groups = df.groupby('genus')
        for genus, group in genus_groups:
            if pd.isna(genus) or genus == '' or len(group) < 3:
                continue
            centroids['genus'][genus] = (
                group['median_L'].median(),
                group['median_a'].median(),
                group['median_b'].median()
            )
        logger.info(f"Computed {len(centroids['genus'])} genus centroids")

        return centroids

    def _compute_distinctiveness(self, df, centroids):
        """
        Add distinctiveness features to DataFrame.

        Args:
            df: DataFrame with per-image features and taxonomic info
            centroids: Dict from _compute_taxonomic_centroids

        Returns:
            DataFrame with added distinctiveness columns
        """
        df = df.copy()

        # Initialize distinctiveness columns
        df['distinctiveness_global'] = np.nan
        df['distinctiveness_family'] = np.nan
        df['distinctiveness_genus'] = np.nan

        for idx, row in df.iterrows():
            image_lab = (row['median_L'], row['median_a'], row['median_b'])

            # Global distinctiveness
            if centroids['global'] is not None:
                df.at[idx, 'distinctiveness_global'] = compute_distinctiveness_from_group(
                    image_lab, centroids['global']
                )

            # Family distinctiveness
            if pd.notna(row['family']) and row['family'] in centroids['family']:
                df.at[idx, 'distinctiveness_family'] = compute_distinctiveness_from_group(
                    image_lab, centroids['family'][row['family']]
                )

            # Genus distinctiveness
            if pd.notna(row['genus']) and row['genus'] in centroids['genus']:
                df.at[idx, 'distinctiveness_genus'] = compute_distinctiveness_from_group(
                    image_lab, centroids['genus'][row['genus']]
                )

        return df

    def extract_all_features(self):
        """
        Run the complete feature extraction pipeline.

        Returns:
            DataFrame with all features
        """
        logger.info("Starting feature extraction...")

        rows = []

        for mask_file, metadata in self.masks_metadata.items():
            # Get colors_visual (saliency-ranked colors)
            colors_visual = metadata.get('colors_visual', [])

            # Extract features
            features = self._compute_per_image_features(colors_visual)

            # Match mask filename to illustration
            # mask_file is like "KW_T_423_1_0020R.tif_mask.png"
            # file_key is like "KW_T_423_1_0020R.tif"
            base_filename = mask_file.replace('_mask.png', '')

            # Look up in illustrations_df
            illus_match = self.illustrations_df[
                self.illustrations_df['file_key'] == base_filename
            ]

            if len(illus_match) == 0:
                logger.warning(f"No illustration match for {mask_file}")
                family = None
                genus = None
                species = None
                deel = None
                illustration_order = None
            else:
                if len(illus_match) > 1:
                    logger.warning(f"Multiple illustration matches for {mask_file}")
                illus_match = illus_match.iloc[0]
                deel = illus_match['Volume']
                illustration_order = illus_match['illustration_order']

                # Now match to taxonomy using illustration_order
                tax_match = self.taxonomy_df[
                    self.taxonomy_df['illustration_order'] == illustration_order
                ]

                if len(tax_match) == 0:
                    logger.warning(
                        f"No taxonomy for order {illustration_order}"
                    )
                    family = None
                    genus = None
                    species = None
                else:
                    tax_match = tax_match.iloc[0]
                    species = tax_match.get('Huidige botanische naam')
                    genus = self._extract_genus(species)

                    # Look up family: try species first, then genus
                    family = self.species_family_map.get(species)
                    if family is None and genus is not None:
                        family = self.genus_family_map.get(genus)

            # Build row
            row = {
                'mask_file': mask_file,
                'image_file': metadata.get('image'),
                'volume': deel,
                'illustration_order': illustration_order,
                'family': family,
                'genus': genus,
                'species': species,
                **features  # Add all extracted features
            }

            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)
        logger.info(f"Extracted features for {len(df)} images")

        # Compute centroids
        centroids = self._compute_taxonomic_centroids(df)

        # Compute distinctiveness
        df = self._compute_distinctiveness(df, centroids)

        logger.info("Feature extraction complete!")

        return df

    def export_to_csv(self, df, output_dir='visualizations'):
        """
        Export feature matrix and taxonomic summary to CSV files.

        Args:
            df: DataFrame from extract_all_features()
            output_dir: Output directory for CSV files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Export full feature matrix
        features_path = output_dir / 'color_features.csv'
        df.to_csv(features_path, index=False)
        logger.info(f"Saved color features to {features_path}")

        # Create taxonomic summary
        summary_rows = []

        # Family-level summary
        family_groups = df.groupby('family')
        for family, group in family_groups:
            if pd.isna(family) or family == '':
                continue
            summary_rows.append({
                'level': 'family',
                'family': family,
                'genus': None,
                'n_images': len(group),
                'centroid_L': group['median_L'].median(),
                'centroid_a': group['median_a'].median(),
                'centroid_b': group['median_b'].median(),
                'mean_distinctiveness_global': group['distinctiveness_global'].mean(),
                'std_distinctiveness_global': group['distinctiveness_global'].std()
            })

        # Genus-level summary
        genus_groups = df.groupby(['family', 'genus'])
        for (family, genus), group in genus_groups:
            if pd.isna(genus) or genus == '':
                continue
            summary_rows.append({
                'level': 'genus',
                'family': family,
                'genus': genus,
                'n_images': len(group),
                'centroid_L': group['median_L'].median(),
                'centroid_a': group['median_a'].median(),
                'centroid_b': group['median_b'].median(),
                'mean_distinctiveness_global': group['distinctiveness_global'].mean(),
                'std_distinctiveness_global': group['distinctiveness_global'].std()
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / 'taxonomic_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved taxonomic summary to {summary_path}")

        return features_path, summary_path


class TaxonomicColorAnalyzer:
    """
    Analyze color variation within and between taxonomic groups.
    Computes dispersion metrics, between-group distances, and identifies overlapping groups.
    """

    def __init__(self, df):
        """
        Initialize with a feature DataFrame.

        Args:
            df: DataFrame from ColorFeatureExtractor.extract_all_features()
                Must have columns: median_L, median_a, median_b, family, genus
        """
        self.df = df.copy()

    def compute_within_group_dispersion(self, group_column='family'):
        """
        For each taxonomic group, compute color dispersion metrics.

        Args:
            group_column: Column to group by ('family' or 'genus')

        Returns:
            DataFrame with columns:
                - [group_column]: taxonomic group name
                - n_images: number of images in group
                - mean_pairwise_distance: average LAB distance between all image colors
                - std_pairwise_distance: standard deviation of distances
                - color_space_volume: volume of convex hull in LAB space (if ≥4 images)
                - centroid_L, centroid_a, centroid_b: group centroid in LAB
        """
        from scipy.spatial.distance import pdist
        from scipy.spatial import ConvexHull

        results = []

        for group_name, group_df in self.df.groupby(group_column):
            if pd.isna(group_name) or group_name == '':
                continue

            # Collect LAB values for all images in group
            lab_values = group_df[['median_L', 'median_a', 'median_b']].dropna().values

            if len(lab_values) < 2:
                # Not enough data for pairwise distances
                results.append({
                    group_column: group_name,
                    'n_images': len(group_df),
                    'mean_pairwise_distance': np.nan,
                    'std_pairwise_distance': np.nan,
                    'color_space_volume': np.nan,
                    'centroid_L': lab_values[0, 0] if len(lab_values) == 1 else np.nan,
                    'centroid_a': lab_values[0, 1] if len(lab_values) == 1 else np.nan,
                    'centroid_b': lab_values[0, 2] if len(lab_values) == 1 else np.nan,
                })
                continue

            # Pairwise distances
            pairwise_dists = pdist(lab_values, metric='euclidean')
            mean_dist = np.mean(pairwise_dists)
            std_dist = np.std(pairwise_dists)

            # Convex hull volume (requires at least 4 points in 3D)
            volume = np.nan
            if len(lab_values) >= 4:
                try:
                    hull = ConvexHull(lab_values)
                    volume = hull.volume
                except Exception:
                    pass  # Degenerate case (e.g., all colors coplanar)

            # Centroid (median for robustness)
            centroid = np.median(lab_values, axis=0)

            results.append({
                group_column: group_name,
                'n_images': len(group_df),
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

        Args:
            group_column: Column to group by ('family' or 'genus')

        Returns:
            DataFrame: pairwise distance matrix between groups (symmetric)
        """
        # Get group centroids
        centroids = {}
        for group_name, group_df in self.df.groupby(group_column):
            if pd.isna(group_name) or group_name == '':
                continue
            lab_values = group_df[['median_L', 'median_a', 'median_b']].dropna().values
            if len(lab_values) > 0:
                centroids[group_name] = np.median(lab_values, axis=0)

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

        Args:
            group_column: Column to group by ('family' or 'genus')
            threshold: LAB distance threshold for considering colors "overlapping"

        Returns:
            List of tuples: (group1, group2, distance) sorted by distance ascending
        """
        between_dists = self.compute_between_group_distances(group_column)

        overlaps = []
        for i, group_i in enumerate(between_dists.index):
            for j, group_j in enumerate(between_dists.columns):
                if i < j:  # Upper triangle only
                    dist = between_dists.iloc[i, j]
                    if dist < threshold and dist > 0:
                        overlaps.append((group_i, group_j, dist))

        return sorted(overlaps, key=lambda x: x[2])


def analyze_taxonomic_colors(df):
    """
    Perform taxonomic-level color analysis.

    Args:
        df: DataFrame from ColorFeatureExtractor.extract_all_features()

    Returns:
        dict with keys:
            - family_dispersion: DataFrame of within-family color metrics
            - genus_dispersion: DataFrame of within-genus color metrics
            - family_distances: DataFrame of between-family distances
            - family_overlaps: List of overlapping families
    """
    analyzer = TaxonomicColorAnalyzer(df)

    return {
        'family_dispersion': analyzer.compute_within_group_dispersion('family'),
        'genus_dispersion': analyzer.compute_within_group_dispersion('genus'),
        'family_distances': analyzer.compute_between_group_distances('family'),
        'family_overlaps': analyzer.identify_color_overlap('family'),
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_lab_space_3d(df, color_by='family', alpha=0.6, output_path=None):
    """
    3D scatter plot of colors in LAB space.

    Args:
        df: DataFrame with median_L, median_a, median_b columns
        color_by: Column name to color points by (e.g., 'family', 'genus')
        alpha: Point transparency
        output_path: If provided, save figure to this path

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Filter to rows with valid LAB and grouping values
    plot_df = df[df[color_by].notna() & df['median_L'].notna()].copy()

    # Color by category
    categories = plot_df[color_by].unique()
    cmap = plt.cm.tab20
    colors = cmap(np.linspace(0, 1, min(len(categories), 20)))

    for i, category in enumerate(categories):
        subset = plot_df[plot_df[color_by] == category]
        ax.scatter(
            subset['median_L'],
            subset['median_a'],
            subset['median_b'],
            c=[colors[i % 20]],
            label=category if len(categories) <= 15 else None,
            alpha=alpha,
            s=30
        )

    ax.set_xlabel('L* (Lightness)')
    ax.set_ylabel('a* (Red-Green)')
    ax.set_zlabel('b* (Blue-Yellow)')

    if len(categories) <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.title(f'Color Distribution in LAB Space (by {color_by})')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved 3D LAB plot to {output_path}")

    return fig


def plot_taxonomic_dispersion(dispersion_df, group_column='family', top_n=20, output_path=None):
    """
    Bar plot of color dispersion by taxonomic group.

    Args:
        dispersion_df: DataFrame from TaxonomicColorAnalyzer.compute_within_group_dispersion()
        group_column: Name of the group column
        top_n: Number of top groups to display
        output_path: If provided, save figure to this path

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    # Filter to groups with valid dispersion and sort
    valid_df = dispersion_df[dispersion_df['mean_pairwise_distance'].notna()]
    dispersion_sorted = valid_df.nlargest(top_n, 'mean_pairwise_distance')

    if len(dispersion_sorted) == 0:
        logger.warning("No valid dispersion data to plot")
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(dispersion_sorted))
    bars = ax.barh(y_pos, dispersion_sorted['mean_pairwise_distance'], color='steelblue')

    # Add error bars if std is available
    if 'std_pairwise_distance' in dispersion_sorted.columns:
        ax.errorbar(
            dispersion_sorted['mean_pairwise_distance'],
            y_pos,
            xerr=dispersion_sorted['std_pairwise_distance'],
            fmt='none',
            color='black',
            capsize=3
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(dispersion_sorted[group_column])
    ax.set_xlabel('Mean Pairwise Color Distance (LAB)')
    ax.set_title(f'Color Dispersion by {group_column.capitalize()} (Top {len(dispersion_sorted)})')

    # Add count annotations
    for i, (idx, row) in enumerate(dispersion_sorted.iterrows()):
        ax.annotate(
            f'n={int(row["n_images"])}',
            xy=(row['mean_pairwise_distance'] + 0.5, i),
            va='center',
            fontsize=8,
            color='gray'
        )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved dispersion plot to {output_path}")

    return fig


def plot_color_features_correlation(df, output_path=None):
    """
    Correlation heatmap of color features.

    Args:
        df: DataFrame with color features
        output_path: If provided, save figure to this path

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    # Select only numeric feature columns (exclude identifiers)
    exclude_cols = ['mask_file', 'image_file', 'volume', 'illustration_order',
                    'family', 'genus', 'species']
    feature_cols = [col for col in df.columns
                    if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]

    if len(feature_cols) < 2:
        logger.warning("Not enough numeric columns for correlation plot")
        return None

    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))

    # Try seaborn, fall back to matplotlib imshow
    try:
        import seaborn as sns
        sns.heatmap(
            corr,
            annot=False,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
    except ImportError:
        # Fallback to matplotlib
        im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.index)))
        ax.set_xticklabels(corr.columns)
        ax.set_yticklabels(corr.index)
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.title('Color Feature Correlations')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved correlation heatmap to {output_path}")

    return fig


def plot_between_group_distances(distance_df, output_path=None):
    """
    Heatmap of between-group color distances.

    Args:
        distance_df: DataFrame from TaxonomicColorAnalyzer.compute_between_group_distances()
        output_path: If provided, save figure to this path

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    if len(distance_df) < 2:
        logger.warning("Not enough groups for distance heatmap")
        return None

    fig, ax = plt.subplots(figsize=(12, 10))

    # Try seaborn, fall back to matplotlib imshow
    try:
        import seaborn as sns
        sns.heatmap(
            distance_df,
            annot=len(distance_df) <= 15,
            fmt='.1f',
            cmap='viridis_r',
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "LAB Distance"},
            ax=ax
        )
    except ImportError:
        # Fallback to matplotlib
        im = ax.imshow(distance_df.values, cmap='viridis_r', aspect='auto')
        ax.set_xticks(np.arange(len(distance_df.columns)))
        ax.set_yticks(np.arange(len(distance_df.index)))
        ax.set_xticklabels(distance_df.columns)
        ax.set_yticklabels(distance_df.index)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('LAB Distance')

    plt.title('Between-Group Color Distances (LAB)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved distance heatmap to {output_path}")

    return fig


def generate_all_visualizations(df, taxonomic_analysis, output_dir='visualizations'):
    """
    Generate all matplotlib visualizations and save to output directory.

    Args:
        df: DataFrame from ColorFeatureExtractor.extract_all_features()
        taxonomic_analysis: Dict from analyze_taxonomic_colors()
        output_dir: Directory to save plots

    Returns:
        List of output file paths
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    output_files = []

    # 1. LAB space 3D plot by family
    try:
        fig = plot_lab_space_3d(df, color_by='family',
                                output_path=output_dir / 'lab_space_by_family.png')
        if fig:
            output_files.append(output_dir / 'lab_space_by_family.png')
            plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to create LAB 3D plot: {e}")

    # 2. Family dispersion bar chart
    try:
        fig = plot_taxonomic_dispersion(
            taxonomic_analysis['family_dispersion'],
            group_column='family',
            output_path=output_dir / 'family_dispersion.png'
        )
        if fig:
            output_files.append(output_dir / 'family_dispersion.png')
            plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to create family dispersion plot: {e}")

    # 3. Genus dispersion bar chart
    try:
        fig = plot_taxonomic_dispersion(
            taxonomic_analysis['genus_dispersion'],
            group_column='genus',
            output_path=output_dir / 'genus_dispersion.png'
        )
        if fig:
            output_files.append(output_dir / 'genus_dispersion.png')
            plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to create genus dispersion plot: {e}")

    # 4. Feature correlation heatmap
    try:
        fig = plot_color_features_correlation(df,
                                              output_path=output_dir / 'feature_correlations.png')
        if fig:
            output_files.append(output_dir / 'feature_correlations.png')
            plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to create correlation heatmap: {e}")

    # 5. Between-family distance heatmap
    try:
        fig = plot_between_group_distances(
            taxonomic_analysis['family_distances'],
            output_path=output_dir / 'family_distances.png'
        )
        if fig:
            output_files.append(output_dir / 'family_distances.png')
            plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to create distance heatmap: {e}")

    logger.info(f"Generated {len(output_files)} visualization files")
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description='Extract color features from segmented plant images'
    )
    parser.add_argument(
        '--masks-metadata',
        type=str,
        default='masks/masks_metadata.json',
        help='Path to masks metadata JSON file'
    )
    parser.add_argument(
        '--taxonomy',
        type=str,
        default='data/Flora Batava Index.xlsx',
        help='Path to Flora Batava taxonomy Excel file'
    )
    parser.add_argument(
        '--meta',
        type=str,
        default='flora_meta.xlsx',
        help='Path to flora_meta.xlsx (filename to order mapping)'
    )
    parser.add_argument(
        '--taxonomy-sheet',
        type=int,
        default=0,
        help='Sheet index in taxonomy file (default: 0 = first sheet)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Output directory for CSV files and plots'
    )
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip generating matplotlib visualizations'
    )

    args = parser.parse_args()

    try:
        # Create extractor
        extractor = ColorFeatureExtractor(
            masks_metadata_path=args.masks_metadata,
            taxonomy_path=args.taxonomy,
            meta_path=args.meta,
            taxonomy_sheet=args.taxonomy_sheet
        )

        # Extract features
        df = extractor.extract_all_features()

        # Export to CSV
        features_path, summary_path = extractor.export_to_csv(df, output_dir=args.output_dir)

        # Run taxonomic analysis
        logger.info("\n=== Running Taxonomic Color Analysis ===")
        taxonomic_analysis = analyze_taxonomic_colors(df)

        # Export taxonomic dispersion data
        output_dir = Path(args.output_dir)
        family_disp_path = output_dir / 'family_dispersion.csv'
        taxonomic_analysis['family_dispersion'].to_csv(family_disp_path, index=False)
        logger.info(f"Saved family dispersion to {family_disp_path}")

        genus_disp_path = output_dir / 'genus_dispersion.csv'
        taxonomic_analysis['genus_dispersion'].to_csv(genus_disp_path, index=False)
        logger.info(f"Saved genus dispersion to {genus_disp_path}")

        family_dist_path = output_dir / 'family_distances.csv'
        taxonomic_analysis['family_distances'].to_csv(family_dist_path)
        logger.info(f"Saved family distances to {family_dist_path}")

        # Log overlapping families
        if taxonomic_analysis['family_overlaps']:
            logger.info(f"\nFamilies with overlapping color spaces (distance < 20):")
            for fam1, fam2, dist in taxonomic_analysis['family_overlaps'][:10]:
                logger.info(f"  {fam1} <-> {fam2}: {dist:.1f}")

        # Generate visualizations
        if not args.skip_plots:
            logger.info("\n=== Generating Visualizations ===")
            plot_files = generate_all_visualizations(df, taxonomic_analysis, args.output_dir)
            for pf in plot_files:
                logger.info(f"  - {pf}")

        logger.info("\n=== Feature Extraction Summary ===")
        logger.info(f"Total images: {len(df)}")
        logger.info(f"Images with taxonomy: {df['family'].notna().sum()}")
        logger.info(f"Unique families: {df['family'].nunique()}")
        logger.info(f"Unique genera: {df['genus'].nunique()}")
        logger.info(f"\nOutput files:")
        logger.info(f"  - {features_path}")
        logger.info(f"  - {summary_path}")
        logger.info(f"  - {family_disp_path}")
        logger.info(f"  - {genus_disp_path}")
        logger.info(f"  - {family_dist_path}")

    except Exception as e:
        logger.error(f"Error during feature extraction: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
