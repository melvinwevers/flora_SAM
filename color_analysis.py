#!/usr/bin/env python3
"""
Color Analysis Tool for Segmented Images

Extracts dominant colors from segmented flora images:
- Extracts top 5 dominant colors using k-means clustering
- Provides RGB, HSL, and LAB color representations
- Filters out background colors (optional)
- Updates metadata with color information
"""

import argparse
import colorsys
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from tqdm import tqdm


def rgb_to_hsl(r, g, b):
    """Convert RGB (0-255) to HSL (H: 0-360, S: 0-100, L: 0-100)"""
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    return (int(h * 360), int(s * 100), int(l * 100))


def rgb_to_lab(r, g, b):
    """
    Convert RGB (0-255) to LAB color space.

    Args:
        r, g, b: RGB values in range 0-255

    Returns:
        Tuple of (L, a, b) where:
        - L: Lightness (0-100)
        - a: Green-Red axis (-128 to 127)
        - b: Blue-Yellow axis (-128 to 127)
    """
    # Normalize RGB to 0-1 range and reshape for skimage
    rgb_normalized = np.array([[[r / 255.0, g / 255.0, b / 255.0]]])
    lab = rgb2lab(rgb_normalized)
    return [float(lab[0, 0, 0]), float(lab[0, 0, 1]), float(lab[0, 0, 2])]


def color_distance(color1, color2):
    """
    Calculate Euclidean distance between two RGB colors.

    Args:
        color1: Tuple of (R, G, B) values
        color2: Tuple of (R, G, B) values

    Returns:
        Distance as a float
    """
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    return np.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)


def color_distance_lab(lab1, lab2):
    """
    Calculate Delta E (CIE76) distance between two LAB colors.
    This is perceptually uniform - equal distances represent equal perceived differences.

    Args:
        lab1: Tuple/list of (L, a, b) values
        lab2: Tuple/list of (L, a, b) values

    Returns:
        Delta E distance as a float
    """
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    return np.sqrt((l1 - l2)**2 + (a1 - a2)**2 + (b1 - b2)**2)


def is_similar_to_background(rgb, background_rgb, tolerance=50):
    """
    Check if a color is similar to the background color.

    Args:
        rgb: Tuple of (R, G, B) values to check
        background_rgb: Background color as (R, G, B) tuple
        tolerance: Maximum distance to consider colors similar (0-441)

    Returns:
        True if the color is similar to the background
    """
    if background_rgb is None:
        return False

    distance = color_distance(rgb, background_rgb)
    return distance <= tolerance


def calculate_perceptual_weight(rgb, hsl, lab, percentage, all_colors_lab):
    """
    Calculate perceptual weight combining frequency, saturation, and contrast.

    This is a heuristic for color palette ranking, NOT true visual salience.

    Args:
        rgb: RGB tuple of the color (kept for backwards compatibility)
        hsl: HSL tuple of the color (H, S, L)
        lab: LAB tuple of the color (L, a, b)
        percentage: Frequency percentage
        all_colors_lab: List of all LAB colors for perceptually accurate contrast calculation

    Returns:
        Perceptual weight score (higher = more perceptually distinctive)
    """
    # Saturation component (0-100) - more saturated colors are more perceptually distinctive
    saturation_score = hsl[1] / 100.0

    # Calculate average perceptual distance to OTHER colors (excluding self)
    if len(all_colors_lab) > 1:
        # Find index of current color
        current_index = None
        for i, existing_lab in enumerate(all_colors_lab):
            # Use numpy.allclose for floating-point comparison
            if np.allclose(existing_lab, lab, atol=0.1):
                current_index = i
                break

        # Get distances to all OTHER colors (exclude current)
        distances = []
        for i, other_lab in enumerate(all_colors_lab):
            if i != current_index:
                dist = color_distance_lab(lab, other_lab)
                distances.append(dist)

        # Average distance (0 if only one color in palette)
        avg_distance = np.mean(distances) if distances else 0

        # Normalize to 0-1 (typical max Delta E in a palette is ~100-150, use 150 for normalization)
        contrast_score = min(avg_distance / 150.0, 1.0)
    else:
        contrast_score = 0

    # Frequency component (already 0-100, normalize to 0-1)
    frequency_score = percentage / 100.0

    # Weighted combination: 40% frequency, 30% saturation, 30% contrast
    # This gives frequency most weight but still values visually distinct colors
    weight = (0.4 * frequency_score) + (0.3 * saturation_score) + (0.3 * contrast_score)

    return weight


def compute_saliency_map(image, mask=None, max_size=512):
    """
    Compute saliency map for an image once (to be reused for all colors).

    Args:
        image: BGR image (OpenCV format)
        mask: Optional binary mask of the segmented object
        max_size: Maximum dimension for saliency computation (downscale if larger)

    Returns:
        Saliency map (2D array, values 0-1) and RGB image
    """
    try:
        # Downscale image for faster saliency computation if needed
        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image_small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            image_small = image

        # Convert to RGB for saliency detection
        image_rgb = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)

        # Try OpenCV contrib saliency module
        if hasattr(cv2, 'saliency'):
            # Create saliency detector using Spectral Residual method
            saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency_detector.computeSaliency(image_rgb)

            if not success:
                raise ValueError("Saliency computation failed")
        else:
            # Fallback: simple edge-based saliency approximation
            # Convert to grayscale
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

            # Compute edges using Sobel
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobelx**2 + sobely**2)

            # Normalize to 0-1
            if edge_magnitude.max() > 0:
                saliency_map = edge_magnitude / edge_magnitude.max()
            else:
                saliency_map = edge_magnitude

            # Smooth the saliency map
            saliency_map = cv2.GaussianBlur(saliency_map, (9, 9), 0)

        # Upscale saliency map back to original size if we downscaled
        if scale < 1.0:
            saliency_map = cv2.resize(saliency_map, (w, h), interpolation=cv2.INTER_LINEAR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Use original full-size image

        # Apply mask if provided
        if mask is not None:
            saliency_map = saliency_map * mask

        return saliency_map, image_rgb

    except Exception as e:
        print(f"Warning: Saliency map computation failed: {e}")
        return None, None


def calculate_saliency_weight(saliency_map, image_rgb, rgb):
    """
    Calculate saliency weight for a specific color using LAB-based color matching.

    Uses LAB color space for perceptually accurate color matching.
    A Delta E of 15 is a "noticeable difference" - good threshold for grouping
    colors from the same K-means cluster.

    Args:
        saliency_map: Precomputed saliency map (2D array, 0-1)
        image_rgb: RGB image
        rgb: RGB tuple of the color to weight

    Returns:
        Saliency weight (0-1, higher = more salient)
    """
    try:
        if saliency_map is None or image_rgb is None:
            return 0.0

        from skimage.color import rgb2lab

        # Convert cluster center to LAB
        rgb_norm = np.array([[[rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0]]])
        lab_center = rgb2lab(rgb_norm)[0, 0]

        # Use Delta E threshold of 15 (noticeable but related colors)
        # This groups pixels from the same K-means cluster together
        delta_e_threshold = 15.0

        # Convert full image to LAB
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Scale LAB values to standard ranges
        # OpenCV LAB: L in 0-255, a in 0-255, b in 0-255
        # Standard LAB: L in 0-100, a in -128-127, b in -128-127
        image_lab[:,:,0] = (image_lab[:,:,0] / 255.0) * 100.0
        image_lab[:,:,1] = image_lab[:,:,1] - 128.0
        image_lab[:,:,2] = image_lab[:,:,2] - 128.0

        # Calculate Delta E (CIE76) for each pixel
        delta_L = (image_lab[:,:,0] - lab_center[0])**2
        delta_a = (image_lab[:,:,1] - lab_center[1])**2
        delta_b = (image_lab[:,:,2] - lab_center[2])**2

        delta_e = np.sqrt(delta_L + delta_a + delta_b)

        # Create binary mask for colors within threshold
        color_mask = (delta_e <= delta_e_threshold).astype(np.uint8)

        # Get mean saliency where this color appears
        if np.any(color_mask):
            saliency_for_color = saliency_map[color_mask > 0]
            mean_saliency = np.mean(saliency_for_color)
            return float(mean_saliency)
        else:
            return 0.0

    except Exception as e:
        print(f"Warning: Saliency weight calculation failed: {e}")
        return 0.0


def extract_dominant_colors(
    image,
    n_colors=5,
    filter_background=True,
    background_color=None,
    background_tolerance=50,
    sample_size=10000,
    calculate_both=False
):
    """
    Extract dominant colors from an image using k-means clustering.

    Args:
        image: BGR image (OpenCV format)
        n_colors: Number of dominant colors to extract
        filter_background: Whether to filter out background colors
        background_color: RGB tuple of background color to filter out
        background_tolerance: Tolerance for background color similarity
        sample_size: Number of pixels to sample (for performance)
        calculate_both: If True, return dict with both frequency and visual importance rankings

    Returns:
        If calculate_both=False: List of color dictionaries (sorted by frequency)
        If calculate_both=True: Dict with 'frequency' and 'visual' keys, each containing color lists
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get only non-zero pixels (masked object)
    # Create a mask of non-black pixels
    mask = np.any(image_rgb != [0, 0, 0], axis=-1)
    pixels = image_rgb[mask]

    if len(pixels) == 0:
        print("Warning: No non-zero pixels found in image")
        return []

    # Sample pixels if there are too many (for performance)
    if len(pixels) > sample_size:
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[indices]

    # Reshape for k-means
    pixels = pixels.reshape(-1, 3)

    # Perform k-means clustering
    # Try to extract more colors initially if we're filtering
    n_clusters = n_colors * 2 if filter_background else n_colors
    # Reduced max_iter for faster processing with minimal quality loss
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5, max_iter=100)
    kmeans.fit(pixels)

    # Get cluster centers (dominant colors) and their frequencies
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    # Calculate percentage for each color
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_pixels = len(labels)

    # Create color data with percentages
    color_data = []
    all_rgb_colors = []
    all_lab_colors = []

    for label, count in zip(unique_labels, counts):
        rgb = tuple(colors[label])
        percentage = (count / total_pixels) * 100

        # Filter background colors if requested
        if filter_background and background_color is not None:
            if is_similar_to_background(rgb, background_color, background_tolerance):
                continue

        hsl = rgb_to_hsl(*rgb)
        lab = rgb_to_lab(*rgb)
        all_rgb_colors.append(rgb)
        all_lab_colors.append(lab)

        color_data.append({
            'rgb': [int(rgb[0]), int(rgb[1]), int(rgb[2])],
            'hsl': [int(hsl[0]), int(hsl[1]), int(hsl[2])],
            'lab': lab,
            'percentage': round(float(percentage), 2),
            'hex': '#{:02x}{:02x}{:02x}'.format(*rgb),
            'rgb_tuple': rgb,  # Keep for visual importance calculation
            'hsl_tuple': hsl,
            'lab_tuple': lab  # Keep for perceptual weight calculation
        })

    # Always calculate visual importance for all colors
    if len(color_data) > 0:
        # Compute saliency map once for all colors (major optimization)
        saliency_map, image_rgb = compute_saliency_map(image)

        for color in color_data:
            # Calculate perceptual weight (heuristic based on color properties)
            # Uses LAB for perceptually accurate contrast calculation
            perceptual = calculate_perceptual_weight(
                color['rgb_tuple'],
                color['hsl_tuple'],
                color['lab_tuple'],
                color['percentage'],
                all_lab_colors
            )
            color['perceptual_weight'] = round(perceptual, 4)

            # Calculate true visual salience using precomputed saliency map
            saliency = calculate_saliency_weight(saliency_map, image_rgb, color['rgb_tuple'])
            color['saliency_weight'] = round(saliency, 4)

            # For backwards compatibility, use saliency as primary
            color['visual_importance'] = round(saliency, 4)

    # If calculate_both, return all three rankings
    if calculate_both:
        # Create three sorted lists
        frequency_sorted = sorted(color_data, key=lambda x: x['percentage'], reverse=True)
        perceptual_sorted = sorted(color_data, key=lambda x: x['perceptual_weight'], reverse=True)
        saliency_sorted = sorted(color_data, key=lambda x: x['saliency_weight'], reverse=True)

        # Clean up temporary fields for all lists
        all_colors = frequency_sorted + perceptual_sorted + saliency_sorted
        for color in all_colors:
            color.pop('rgb_tuple', None)
            color.pop('hsl_tuple', None)

        return {
            'frequency': frequency_sorted[:n_colors],
            'perceptual': perceptual_sorted[:n_colors],
            'saliency': saliency_sorted[:n_colors]
        }
    else:
        # Default: sort by percentage (most dominant first)
        color_data.sort(key=lambda x: x['percentage'], reverse=True)

        # Clean up temporary fields
        for color in color_data:
            color.pop('rgb_tuple', None)
            color.pop('hsl_tuple', None)

        # Return only top n_colors
        return color_data[:n_colors]


class ColorAnalyzer:
    def __init__(
        self,
        mask_output_dir,
        n_colors=5,
        filter_background=True,
        background_tolerance=50,
        force=False
    ):
        """
        Initialize Color Analyzer.

        Args:
            mask_output_dir: Directory containing segmented images and metadata
            n_colors: Number of dominant colors to extract per image
            filter_background: Whether to filter out background colors
            background_tolerance: Color distance tolerance for background filtering
            force: Force reprocessing of all images
        """
        self.mask_output_dir = Path(mask_output_dir)
        self.n_colors = n_colors
        self.filter_background = filter_background
        self.background_tolerance = background_tolerance
        self.force = force

        # Load existing metadata
        self.metadata_file = self.mask_output_dir / "masks_metadata.json"
        self.masks_metadata = self.load_metadata()

        if not self.masks_metadata:
            raise ValueError(f"No metadata found at {self.metadata_file}")

        print(f"Loaded metadata with {len(self.masks_metadata)} entries")

    def load_metadata(self):
        """Load existing metadata file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")
                return {}
        return {}

    def save_metadata(self):
        """Save updated metadata to JSON file"""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.masks_metadata, f, indent=2)
            print(f"✓ Metadata saved to {self.metadata_file}")
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def process_all_images(self):
        """Process all segmented images and extract colors"""
        print(f"\nProcessing {len(self.masks_metadata)} segmented images...")
        print(f"Extracting top {self.n_colors} colors per image")
        print("Computing both frequency and visual importance rankings")
        if self.filter_background:
            print(f"Background filtering: enabled (tolerance: {self.background_tolerance})")
        else:
            print("Background filtering: disabled")
        print()

        processed = 0
        skipped = 0
        errors = 0

        for mask_key, entry in tqdm(self.masks_metadata.items(), desc="Analyzing colors"):
            # Skip if colors already extracted with new three-ranking system (unless --force)
            if not self.force:
                # Check for all three ranking lists and the new weight fields
                has_new_format = False
                if ('colors_frequency' in entry and entry['colors_frequency'] and
                    'colors_perceptual' in entry and entry['colors_perceptual'] and
                    'colors_saliency' in entry and entry['colors_saliency']):
                    # Verify the colors have the new weight fields
                    first_freq = entry['colors_frequency'][0] if entry['colors_frequency'] else {}
                    first_perc = entry['colors_perceptual'][0] if entry['colors_perceptual'] else {}
                    first_sal = entry['colors_saliency'][0] if entry['colors_saliency'] else {}
                    has_new_format = ('perceptual_weight' in first_perc and
                                      'saliency_weight' in first_sal)

                if has_new_format:
                    skipped += 1
                    continue

            # Get segmented image path
            segmented_file = entry.get('segmented_file')
            if not segmented_file:
                print(f"Warning: No segmented_file in entry {mask_key}")
                errors += 1
                continue

            segmented_path = self.mask_output_dir / segmented_file

            if not segmented_path.exists():
                print(f"Warning: Segmented image not found: {segmented_path}")
                errors += 1
                continue

            try:
                # Load segmented image
                image = cv2.imread(str(segmented_path))

                if image is None:
                    print(f"Warning: Could not load image {segmented_path}")
                    errors += 1
                    continue

                # Get background color from metadata if available
                background_color = entry.get('background_color_rgb')
                if background_color and isinstance(background_color, list):
                    background_color = tuple(background_color)

                # Extract colors (both ranking methods)
                color_results = extract_dominant_colors(
                    image,
                    n_colors=self.n_colors,
                    filter_background=self.filter_background,
                    background_color=background_color,
                    background_tolerance=self.background_tolerance,
                    calculate_both=True
                )

                # Update metadata with all three rankings
                entry['colors_frequency'] = color_results['frequency']
                entry['colors_perceptual'] = color_results['perceptual']
                entry['colors_saliency'] = color_results['saliency']
                # Keep 'colors' as default (frequency) for backwards compatibility
                entry['colors'] = color_results['frequency']
                # Keep colors_visual for backwards compatibility (use saliency)
                entry['colors_visual'] = color_results['saliency']
                entry['n_colors_extracted'] = len(color_results['frequency'])

                processed += 1

            except Exception as e:
                print(f"Error processing {segmented_file}: {e}")
                errors += 1

        # Save updated metadata
        self.save_metadata()

        # Print summary
        print(f"\n{'='*60}")
        print(f"COLOR ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Processed: {processed}")
        print(f"Skipped (already done): {skipped}")
        print(f"Errors: {errors}")
        print(f"Total: {len(self.masks_metadata)}")
        print(f"{'='*60}")

    def show_color_stats(self):
        """Show statistics about extracted colors"""
        total_colors = 0
        images_with_colors = 0

        for entry in self.masks_metadata.values():
            # Check for both new and old format
            if 'colors_frequency' in entry and entry['colors_frequency']:
                images_with_colors += 1
                total_colors += len(entry['colors_frequency'])
            elif 'colors' in entry and entry['colors']:
                images_with_colors += 1
                total_colors += len(entry['colors'])

        if images_with_colors > 0:
            avg_colors = total_colors / images_with_colors
            print(f"\nColor extraction statistics:")
            print(f"  Images with colors: {images_with_colors}/{len(self.masks_metadata)}")
            print(f"  Total colors extracted: {total_colors} (per ranking method)")
            print(f"  Average colors per image: {avg_colors:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract dominant colors from segmented images"
    )
    parser.add_argument(
        "--masks-dir",
        default="masks",
        help="Directory containing masks and segmented images"
    )
    parser.add_argument(
        "--n-colors",
        type=int,
        default=5,
        help="Number of dominant colors to extract (default: 5)"
    )
    parser.add_argument(
        "--no-filter-background",
        action="store_true",
        help="Don't filter out background colors"
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=50,
        help="Color distance tolerance for background filtering (default: 50)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show color extraction statistics only (don't process)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of all images (skip the skip check)"
    )

    args = parser.parse_args()

    try:
        analyzer = ColorAnalyzer(
            mask_output_dir=args.masks_dir,
            n_colors=args.n_colors,
            filter_background=not args.no_filter_background,
            background_tolerance=args.tolerance,
            force=args.force
        )

        if args.stats:
            analyzer.show_color_stats()
        else:
            analyzer.process_all_images()
            analyzer.show_color_stats()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
