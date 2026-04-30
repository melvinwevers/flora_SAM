#!/usr/bin/env python3
"""
Color Analysis Tool for Segmented Images

Extracts dominant colors from segmented flora images:
- Extracts top 5 dominant colors using k-means clustering
- Provides RGB, HSL, and LAB color representations
- Filters out background colors (optional)
- Calculates 3 color rankings: frequency, saliency, chroma
- Updates metadata with color information
"""

import argparse
import colorsys
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from skimage.color import rgb2lab
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def crop_uniform_borders(image, threshold=20, min_border_width=5):
    """
    Crop uniform color borders from image edges.

    Detects and removes black/white borders that may have been included
    in SAM segmentation due to scan artifacts.

    Args:
        image: RGB/RGBA image array
        threshold: Color variance threshold (0-255) to consider uniform
        min_border_width: Minimum pixels to crop (avoids tiny crops)

    Returns:
        Cropped image array
    """
    if len(image.shape) == 3 and image.shape[2] == 4:
        # RGBA - use RGB channels for analysis
        img_rgb = image[:, :, :3]
    else:
        img_rgb = image

    h, w = img_rgb.shape[:2]

    # Analyze border regions (10% of image on each side)
    border_size = max(min(h, w) // 20, min_border_width)

    # Top border
    top_region = img_rgb[:border_size, :]
    top_std = np.std(top_region)
    crop_top = border_size if top_std < threshold else 0

    # Bottom border
    bottom_region = img_rgb[-border_size:, :]
    bottom_std = np.std(bottom_region)
    crop_bottom = border_size if bottom_std < threshold else 0

    # Left border
    left_region = img_rgb[:, :border_size]
    left_std = np.std(left_region)
    crop_left = border_size if left_std < threshold else 0

    # Right border
    right_region = img_rgb[:, -border_size:]
    right_std = np.std(right_region)
    crop_right = border_size if right_std < threshold else 0

    # Only crop if we found uniform borders
    if crop_top > 0 or crop_bottom > 0 or crop_left > 0 or crop_right > 0:
        # Ensure we don't crop entire image
        new_h = h - crop_top - crop_bottom
        new_w = w - crop_left - crop_right

        if new_h > h // 2 and new_w > w // 2:
            cropped = image[crop_top:h-crop_bottom if crop_bottom > 0 else h,
                          crop_left:w-crop_right if crop_right > 0 else w]
            return cropped

    return image


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
    n_colors=10,
    filter_background=True,
    background_color=None,
    background_tolerance=50,
    sample_size=10000,
    calculate_both=False,
    crop_borders=True,
    filter_extremes=True
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
        crop_borders: Whether to crop uniform borders before analysis
        filter_extremes: Filter out very dark (black) and very bright (white) colors

    Returns:
        If calculate_both=False: List of color dictionaries (sorted by frequency)
        If calculate_both=True: Dict with 'frequency' and 'visual' keys, each containing color lists
    """
    # Crop uniform borders (e.g., black scan borders) if requested
    if crop_borders:
        image = crop_uniform_borders(image)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get only non-zero pixels (masked object)
    # Create a mask of non-black pixels
    mask = np.any(image_rgb != [0, 0, 0], axis=-1)
    pixels = image_rgb[mask]

    if len(pixels) == 0:
        print("Warning: No non-zero pixels found in image")
        empty = {'frequency': [], 'saliency': [], 'chroma': []}
        return empty if calculate_both else []

    total_pixels_full = len(pixels)

    # --- Chroma pass: K-means on only chromatic (non-neutral) pixels ---
    # Filters to LAB chroma = √(a²+b²) > 8, removing paper, beige, and
    # neutral grays. Lowered from 15 to 8 to capture pastel blues/pinks
    # in historical watercolor illustrations. This makes rare coloured
    # features (e.g. 0.3% terracotta flowers, pale blue petals) a
    # significant fraction of the remaining pixels so K-means reliably
    # allocates them their own cluster.
    pixels_lab_all = rgb2lab(
        (pixels.astype(np.float32) / 255.0).reshape(-1, 1, 3)
    ).reshape(-1, 3)
    chroma_all = np.sqrt(pixels_lab_all[:, 1] ** 2 + pixels_lab_all[:, 2] ** 2)
    chromatic_mask = chroma_all > 8.0  # Lowered from 15 to capture pastels
    chromatic_pixels = pixels[chromatic_mask]

    if len(chromatic_pixels) >= n_colors:
        chroma_sample_size = min(len(chromatic_pixels), sample_size)
        chroma_idx = np.random.choice(
            len(chromatic_pixels), chroma_sample_size, replace=False
        )
        chroma_sample = chromatic_pixels[chroma_idx].reshape(-1, 3)
        n_chroma_clusters = min(n_colors * 2, len(chromatic_pixels))
        chroma_kmeans = MiniBatchKMeans(
            n_clusters=n_chroma_clusters, random_state=42, n_init=3, max_iter=100
        )
        chroma_kmeans.fit(chroma_sample)
        chroma_centers = chroma_kmeans.cluster_centers_.astype(int)
        # Count across ALL chromatic pixels for real image percentages
        chroma_diffs = (
            chromatic_pixels[:, np.newaxis, :].astype(float)
            - chroma_centers[np.newaxis, :, :]
        )
        chroma_assign = np.sqrt((chroma_diffs ** 2).sum(axis=2)).argmin(axis=1)
        chroma_counts = np.zeros(len(chroma_centers), dtype=int)
        for lbl, cnt in zip(*np.unique(chroma_assign, return_counts=True)):
            chroma_counts[lbl] = cnt
    else:
        chroma_centers = None

    # --- Regular uniform sample for frequency / saliency ---
    if len(pixels) > sample_size:
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[indices]

    # Reshape for k-means
    pixels = pixels.reshape(-1, 3)

    # Perform k-means clustering
    # Try to extract more colors initially if we're filtering
    n_clusters = n_colors * 2 if filter_background else n_colors
    # Reduced iterations for speed with minimal quality loss
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=100)
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

        # Filter extreme colors (very dark/very bright) that are likely artifacts
        if filter_extremes:
            # Filter very dark colors (likely scan borders/shadows)
            if all(c < 30 for c in rgb):  # Near black
                continue
            # Filter very bright colors (likely paper/background)
            if all(c > 240 for c in rgb):  # Near white
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
            'rgb_tuple': rgb,  # Keep for saliency calculation
        })

    # Calculate saliency for all colors
    if len(color_data) > 0:
        # Compute saliency map once for all colors (major optimization)
        saliency_map, image_rgb = compute_saliency_map(image)

        for color in color_data:
            # Calculate true visual salience using precomputed saliency map
            saliency = calculate_saliency_weight(saliency_map, image_rgb, color['rgb_tuple'])
            color['saliency_weight'] = round(saliency, 4)

            # For backwards compatibility, use saliency as primary
            color['visual_importance'] = round(saliency, 4)

    # If calculate_both, return all rankings
    if calculate_both:
        # Three sorted lists from regular K-means
        frequency_sorted = sorted(
            color_data, key=lambda x: x['percentage'], reverse=True
        )
        saliency_sorted = sorted(
            color_data, key=lambda x: x['saliency_weight'], reverse=True
        )

        for color in (frequency_sorted + saliency_sorted):
            color.pop('rgb_tuple', None)

        # Chroma ranking: K-means on LAB-chroma > 15 pixels only.
        # Neutral paper/beige/gray is removed before clustering so rare
        # coloured features (small flowers, berries) are never swamped by
        # the dominant background and always form their own cluster.
        # Sorted by raw chroma (vividness) = √(a² + b²) to prioritize
        # the most vivid colors, which reveals what the 18th-century
        # illustrator emphasized with saturated watercolor pigments.
        if chroma_centers is not None:
            chroma_color_data = []
            for i, rgb in enumerate(chroma_centers):
                rgb = tuple(int(v) for v in rgb)
                count = int(chroma_counts[i])
                pct = round(count / total_pixels_full * 100, 2)

                if filter_extremes:
                    if all(c < 30 for c in rgb):
                        continue
                    if all(c > 240 for c in rgb):
                        continue

                hsl = rgb_to_hsl(*rgb)
                lab = rgb_to_lab(*rgb)
                center_chroma = (lab[1] ** 2 + lab[2] ** 2) ** 0.5

                chroma_color_data.append({
                    'rgb': list(rgb),
                    'hsl': [int(hsl[0]), int(hsl[1]), int(hsl[2])],
                    'lab': lab,
                    'percentage': pct,
                    'hex': '#{:02x}{:02x}{:02x}'.format(*rgb),
                    'chroma': round(center_chroma, 2),
                })

            chroma_sorted = sorted(
                chroma_color_data, key=lambda x: x['chroma'], reverse=True
            )[:n_colors]
        else:
            chroma_sorted = frequency_sorted[:n_colors]

        return {
            'frequency': frequency_sorted[:n_colors],
            'saliency': saliency_sorted[:n_colors],
            'chroma': chroma_sorted,
        }
    else:
        # Default: sort by percentage (most dominant first)
        color_data.sort(key=lambda x: x['percentage'], reverse=True)

        # Clean up temporary fields
        for color in color_data:
            color.pop('rgb_tuple', None)

        # Return only top n_colors
        return color_data[:n_colors]


def _process_single_image(args):
    """Module-level worker for multiprocessing.Pool (must be picklable)."""
    (mask_key, segmented_path, background_color,
     n_colors, filter_background, background_tolerance, crop_borders) = args

    # One OpenCV thread per worker avoids CPU contention across processes
    cv2.setNumThreads(1)

    try:
        image = cv2.imread(str(segmented_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return (mask_key, None, f"Could not load {segmented_path}")

        # Handle alpha channel for RGBA images
        if image.ndim == 3 and image.shape[2] == 4:
            alpha_channel = image[:, :, 3]
            image[alpha_channel < 10] = 0
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        color_results = extract_dominant_colors(
            image,
            n_colors=n_colors,
            filter_background=filter_background,
            background_color=background_color,
            background_tolerance=background_tolerance,
            calculate_both=True,
            crop_borders=crop_borders,
        )

        return (mask_key, color_results, None)
    except Exception as e:
        return (mask_key, None, str(e))


class ColorAnalyzer:
    def __init__(
        self,
        mask_output_dir,
        n_colors=10,
        filter_background=True,
        background_tolerance=50,
        force=False,
        crop_borders=True,
        n_jobs=None
    ):
        """Initialize Color Analyzer.

        Args:
            mask_output_dir: Directory containing segmented images and metadata
            n_colors: Number of dominant colors to extract per image
            filter_background: Whether to filter out background colors
            background_tolerance: Color distance tolerance for background filtering
            force: Force reprocessing of all images
            crop_borders: Crop uniform color borders before analysis
            n_jobs: Number of parallel jobs (None = use all CPUs)
        """
        self.mask_output_dir = Path(mask_output_dir)
        self.n_colors = n_colors
        self.filter_background = filter_background
        self.background_tolerance = background_tolerance
        self.crop_borders = crop_borders
        self.force = force
        self.n_jobs = n_jobs if n_jobs else cpu_count()
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
        if self.n_jobs > 1:
            print(f"Using {self.n_jobs} parallel workers (sequential mode for stability)")
        print()

        processed = 0
        skipped = 0
        errors = 0
        save_interval = 200  # save progress every N completed images

        # Build work list (skip already-done entries)
        work_items = []
        for mask_key, entry in self.masks_metadata.items():
            if not self.force:
                # Check if all three color rankings exist
                has_all_rankings = (
                    entry.get('colors_frequency')
                    and entry.get('colors_saliency')
                    and entry.get('colors_chroma')
                    # Verify chroma format has 'chroma' field
                    and 'chroma' in (entry['colors_chroma'][0]
                                     if entry.get('colors_chroma') else {})
                )
                if has_all_rankings:
                    skipped += 1
                    continue

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

            background_color = entry.get('background_color_rgb')
            if background_color and isinstance(background_color, list):
                background_color = tuple(background_color)

            work_items.append((
                mask_key, segmented_path, background_color,
                self.n_colors, self.filter_background,
                self.background_tolerance, self.crop_borders,
            ))

        print(f"Work items to process: {len(work_items)}  (skipped: {skipped})")

        with Pool(processes=self.n_jobs) as pool:
            for mask_key, color_results, error in tqdm(
                pool.imap_unordered(_process_single_image, work_items),
                total=len(work_items),
                desc="Analyzing colors",
            ):
                if error or not isinstance(color_results, dict) or not color_results.get('frequency'):
                    print(f"\nError processing {mask_key}: {error}")
                    errors += 1
                    continue

                entry = self.masks_metadata[mask_key]
                entry['colors_frequency'] = color_results['frequency']
                entry['colors_saliency'] = color_results['saliency']
                entry['colors_chroma'] = color_results['chroma']
                entry['colors'] = color_results['frequency']
                entry['colors_visual'] = color_results['saliency']
                entry['n_colors_extracted'] = len(color_results['frequency'])
                processed += 1

                if processed % save_interval == 0:
                    self.save_metadata()
                    print(f"\n  [checkpoint] saved {processed} / {len(work_items)}")

        # Final save
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
        default=10,
        help="Number of dominant colors to extract (default: 10)"
    )
    parser.add_argument(
        "--no-filter-background",
        action="store_true",
        help="Don't filter out background colors"
    )
    parser.add_argument(
        "--no-crop-borders",
        action="store_true",
        help="Don't crop uniform borders (black/white scan artifacts)"
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
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: use all CPUs)"
    )

    args = parser.parse_args()

    try:
        analyzer = ColorAnalyzer(
            mask_output_dir=args.masks_dir,
            n_colors=args.n_colors,
            filter_background=not args.no_filter_background,
            background_tolerance=args.tolerance,
            force=args.force,
            crop_borders=not args.no_crop_borders,
            n_jobs=args.jobs
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
