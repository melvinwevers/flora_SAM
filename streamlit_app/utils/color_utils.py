"""
Color bucketing and analysis utilities.
"""
import colorsys
from typing import Tuple, Dict

# Define color buckets with representative colors and ranges
COLOR_BUCKETS = {
    'Red': {
        'hex': '#DC143C',
        'name': 'Red',
        'hue_range': [(345, 15)],  # Wraps around 0
    },
    'Orange': {
        'hex': '#FF8C00',
        'name': 'Orange',
        'hue_range': [(15, 45)],
    },
    'Yellow': {
        'hex': '#FFD700',
        'name': 'Yellow',
        'hue_range': [(45, 70)],
    },
    'Green': {
        'hex': '#228B22',
        'name': 'Green',
        'hue_range': [(70, 165)],
    },
    'Cyan': {
        'hex': '#00CED1',
        'name': 'Cyan',
        'hue_range': [(165, 195)],
    },
    'Blue': {
        'hex': '#1E90FF',
        'name': 'Blue',
        'hue_range': [(195, 255)],
    },
    'Purple': {
        'hex': '#9370DB',
        'name': 'Purple',
        'hue_range': [(255, 290)],
    },
    'Magenta': {
        'hex': '#C71585',
        'name': 'Magenta',
        'hue_range': [(290, 345)],
    },
    'Brown': {
        'hex': '#8B4513',
        'name': 'Brown',
        'saturation_range': (0.2, 0.7),
        'value_range': (0.2, 0.5),
    },
    'White': {
        'hex': '#F5F5F5',
        'name': 'White',
        'saturation_range': (0, 0.15),
        'value_range': (0.85, 1.0),
    },
    'Gray': {
        'hex': '#808080',
        'name': 'Gray',
        'saturation_range': (0, 0.15),
        'value_range': (0.3, 0.85),
    },
    'Black': {
        'hex': '#2F4F4F',
        'name': 'Black',
        'saturation_range': (0, 0.15),
        'value_range': (0, 0.3),
    },
}


def hex_to_hsv(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color to HSV.

    Args:
        hex_color: Hex color string (e.g., '#FF5733')

    Returns:
        Tuple of (hue [0-360], saturation [0-1], value [0-1])
    """
    # Remove # if present
    hex_color = hex_color.lstrip('#')

    # Convert to RGB
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0

    # Convert to HSV
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # Convert hue to degrees
    h = h * 360

    return h, s, v


def categorize_color(hex_color: str) -> str:
    """Categorize a hex color into a predefined bucket.

    Args:
        hex_color: Hex color string (e.g., '#FF5733')

    Returns:
        Bucket name (e.g., 'Red', 'Green', etc.)
    """
    if not hex_color or str(hex_color) == 'nan':
        return 'Unknown'

    try:
        h, s, v = hex_to_hsv(hex_color)
    except:
        return 'Unknown'

    # Check achromatic colors first (low saturation)
    if s < 0.15:
        if v > 0.85:
            return 'White'
        elif v < 0.3:
            return 'Black'
        else:
            return 'Gray'

    # Check brown (low saturation, mid-low value)
    if 0.2 <= s <= 0.7 and 0.2 <= v <= 0.5:
        return 'Brown'

    # Check chromatic colors by hue
    for bucket_name, bucket_info in COLOR_BUCKETS.items():
        if 'hue_range' not in bucket_info:
            continue

        for hue_min, hue_max in bucket_info['hue_range']:
            # Handle wrap-around for red
            if hue_min > hue_max:
                if h >= hue_min or h <= hue_max:
                    return bucket_name
            else:
                if hue_min <= h <= hue_max:
                    return bucket_name

    return 'Unknown'


def get_color_bucket_info(bucket_name: str) -> Dict:
    """Get information about a color bucket.

    Args:
        bucket_name: Name of the bucket

    Returns:
        Dictionary with bucket information
    """
    return COLOR_BUCKETS.get(bucket_name, {
        'hex': '#CCCCCC',
        'name': bucket_name
    })

def get_group_color_palette(hex_colors, top_n=8):
    """
    Get dominant colors from a group, binned by hue to avoid duplicates.

    Groups colors by hue bucket, then returns the most frequent color from each bucket.
    This prevents getting 5 shades of green and instead shows diversity.

    Args:
        hex_colors: List of hex color strings (e.g., ['#FF5733', '#28A745', ...])
        top_n: Maximum number of colors to return

    Returns:
        List of dicts with 'hex', 'bucket', and 'count' keys, sorted by frequency
    """
    if not hex_colors:
        return []

    # Bin colors by hue bucket
    bucket_colors = {}

    for hex_color in hex_colors:
        if not hex_color or str(hex_color) == 'nan':
            continue

        bucket = categorize_color(hex_color)

        if bucket not in bucket_colors:
            bucket_colors[bucket] = []
        bucket_colors[bucket].append(hex_color)

    # For each bucket, get the most frequent color
    palette = []
    for bucket, colors in bucket_colors.items():
        # Count frequency of each unique color in this bucket
        color_counts = {}
        for color in colors:
            color_counts[color] = color_counts.get(color, 0) + 1

        # Get most frequent
        most_frequent_hex = max(color_counts, key=color_counts.get)
        count = color_counts[most_frequent_hex]

        palette.append({
            'hex': most_frequent_hex,
            'bucket': bucket,
            'count': count
        })

    # Sort by frequency (descending)
    palette.sort(key=lambda x: x['count'], reverse=True)

    # Return top N
    return palette[:top_n]
