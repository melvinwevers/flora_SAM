#!/usr/bin/env python3
"""
Color Visualization Tool for Flora Analysis

Creates interactive visualizations of extracted plant colors:
- 3D color space explorer with thumbnail previews
- Plant color grid
"""

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_color_data(metadata_file, masks_dir):
    """
    Load color data from metadata file (both frequency and visual importance rankings).

    Returns:
        Tuple of (df_frequency, df_visual) DataFrames with columns:
        plant_id, color_idx, rgb, hsl, hex, percentage, segmented_path
    """
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    masks_path = Path(masks_dir)
    rows_freq = []
    rows_visual = []

    for mask_key, entry in metadata.items():
        # Try new format first (with both rankings)
        has_both = 'colors_frequency' in entry and 'colors_visual' in entry

        if not has_both and ('colors' not in entry or not entry['colors']):
            continue

        plant_id = entry['image']
        segmented_file = entry.get('segmented_file', '')
        segmented_path = str(masks_path / segmented_file) if segmented_file else ''

        # Process frequency-based colors
        colors_freq = entry.get('colors_frequency', entry.get('colors', []))
        for idx, color in enumerate(colors_freq):
            rows_freq.append({
                'plant_id': plant_id,
                'plant_short': plant_id.split('.')[0],
                'color_idx': idx,
                'r': color['rgb'][0],
                'g': color['rgb'][1],
                'b': color['rgb'][2],
                'h': color['hsl'][0],
                's': color['hsl'][1],
                'l': color['hsl'][2],
                'hex': color['hex'],
                'percentage': color['percentage'],
                'visual_importance': color.get('visual_importance', 0),
                'rgb_str': f"rgb({color['rgb'][0]},{color['rgb'][1]},{color['rgb'][2]})",
                'segmented_path': segmented_path
            })

        # Process visual importance-based colors
        colors_visual = entry.get('colors_visual', entry.get('colors', []))
        for idx, color in enumerate(colors_visual):
            rows_visual.append({
                'plant_id': plant_id,
                'plant_short': plant_id.split('.')[0],
                'color_idx': idx,
                'r': color['rgb'][0],
                'g': color['rgb'][1],
                'b': color['rgb'][2],
                'h': color['hsl'][0],
                's': color['hsl'][1],
                'l': color['hsl'][2],
                'hex': color['hex'],
                'percentage': color['percentage'],
                'visual_importance': color.get('visual_importance', 0),
                'rgb_str': f"rgb({color['rgb'][0]},{color['rgb'][1]},{color['rgb'][2]})",
                'segmented_path': segmented_path
            })

    return pd.DataFrame(rows_freq), pd.DataFrame(rows_visual)


def create_thumbnail_base64(image_path, max_size=200):
    """
    Create a base64-encoded thumbnail of an image.

    Args:
        image_path: Path to the image
        max_size: Maximum dimension (width or height) in pixels

    Returns:
        Base64-encoded image string or None if image not found
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        # Resize while maintaining aspect ratio
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Encode to PNG
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Warning: Could not create thumbnail for {image_path}: {e}")
        return None


def generate_thumbnails_for_plants(df, max_size=200):
    """
    Generate thumbnails for unique plants and add to dataframe.

    Args:
        df: DataFrame with segmented_path column
        max_size: Maximum thumbnail dimension

    Returns:
        DataFrame with thumbnail_base64 column added
    """
    # Get unique plant paths
    unique_plants = df[['plant_id', 'segmented_path']].drop_duplicates()

    # Generate thumbnails
    thumbnails = {}
    print("Generating thumbnails...")
    for _, row in unique_plants.iterrows():
        if row['segmented_path']:
            thumb = create_thumbnail_base64(row['segmented_path'], max_size)
            thumbnails[row['plant_id']] = thumb

    # Add to dataframe
    df['thumbnail_base64'] = df['plant_id'].map(thumbnails)

    return df


def create_3d_color_space(df_freq, df_visual, sample_size=1000):
    """
    Create 3D scatter plot in HSL color space with thumbnail previews.
    Shows clustering of plants by color with toggle between frequency and visual importance.
    """
    # Sample if too many points
    if len(df_freq) > sample_size:
        df_freq_sample = df_freq.sample(n=sample_size, random_state=42)
        df_visual_sample = df_visual.sample(n=sample_size, random_state=42)
    else:
        df_freq_sample = df_freq.copy()
        df_visual_sample = df_visual.copy()

    # Create frequency trace
    customdata_freq = []
    for _, row in df_freq_sample.iterrows():
        customdata_freq.append([
            row['plant_short'],
            row['hex'],
            row['h'],
            row['s'],
            row['l'],
            row['percentage'],
            row.get('thumbnail_base64', '')
        ])

    # Create visual importance trace
    customdata_visual = []
    for _, row in df_visual_sample.iterrows():
        customdata_visual.append([
            row['plant_short'],
            row['hex'],
            row['h'],
            row['s'],
            row['l'],
            row['percentage'],
            row.get('thumbnail_base64', '')
        ])

    fig = go.Figure(data=[
        # Frequency trace (visible by default)
        go.Scatter3d(
            name='Frequency',
            x=df_freq_sample['h'],
            y=df_freq_sample['s'],
            z=df_freq_sample['l'],
            mode='markers',
            marker=dict(
                size=df_freq_sample['percentage'] / 2,
                color=df_freq_sample['hex'],
                line=dict(color='white', width=0.5),
                opacity=0.8
            ),
            customdata=customdata_freq,
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>' +
                'Color: %{customdata[1]}<br>' +
                'H: %{customdata[2]}° | S: %{customdata[3]}% | L: %{customdata[4]}%<br>' +
                'Dominance: %{customdata[5]:.1f}%<br>' +
                '<extra></extra>'
            ),
            visible=True
        ),
        # Visual importance trace (hidden by default)
        go.Scatter3d(
            name='Visual Importance',
            x=df_visual_sample['h'],
            y=df_visual_sample['s'],
            z=df_visual_sample['l'],
            mode='markers',
            marker=dict(
                size=df_visual_sample['percentage'] / 2,
                color=df_visual_sample['hex'],
                line=dict(color='white', width=0.5),
                opacity=0.8
            ),
            customdata=customdata_visual,
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>' +
                'Color: %{customdata[1]}<br>' +
                'H: %{customdata[2]}° | S: %{customdata[3]}% | L: %{customdata[4]}%<br>' +
                'Dominance: %{customdata[5]:.1f}%<br>' +
                '<extra></extra>'
            ),
            visible=False
        )
    ])

    fig.update_layout(
        title='3D Color Space (HSL) - Plant Colors Clustering (Hover to see thumbnails)',
        scene=dict(
            xaxis_title='Hue (0-360°)',
            yaxis_title='Saturation (0-100%)',
            zaxis_title='Lightness (0-100%)',
        ),
        height=900,
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": [True, False]}],
                        label="Frequency",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True]}],
                        label="Visual Importance",
                        method="update"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ]
    )

    # Add custom JavaScript to show thumbnail on hover
    # This creates a floating div that displays the thumbnail
    custom_js = """
    <div id="thumbnail-preview" style="
        position: fixed;
        top: 20px;
        right: 20px;
        width: 250px;
        background: white;
        border: 2px solid #ccc;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        display: none;
        z-index: 10000;
    ">
        <div id="thumbnail-title" style="font-weight: bold; margin-bottom: 10px; font-size: 14px;"></div>
        <img id="thumbnail-image" style="width: 100%; border-radius: 4px;" />
        <div id="thumbnail-info" style="margin-top: 10px; font-size: 12px; color: #666;"></div>
    </div>
    <script>
        // Wait for plot to be fully rendered
        window.addEventListener('load', function() {
            setTimeout(function() {
                var plots = document.getElementsByClassName('plotly-graph-div');
                if (plots.length === 0) {
                    console.error('No plotly graph div found');
                    return;
                }
                var plot = plots[0];
                var thumbnailDiv = document.getElementById('thumbnail-preview');
                var thumbnailTitle = document.getElementById('thumbnail-title');
                var thumbnailImage = document.getElementById('thumbnail-image');
                var thumbnailInfo = document.getElementById('thumbnail-info');

                plot.on('plotly_hover', function(data) {
                    console.log('Hover event triggered', data);
                    if (data.points && data.points[0] && data.points[0].customdata) {
                        var cd = data.points[0].customdata;
                        var plantName = cd[0];
                        var colorHex = cd[1];
                        var hue = cd[2];
                        var sat = cd[3];
                        var light = cd[4];
                        var perc = cd[5];
                        var thumbnail = cd[6];

                        if (thumbnail) {
                            thumbnailTitle.textContent = plantName;
                            thumbnailImage.src = thumbnail;
                            thumbnailInfo.innerHTML =
                                'Color: ' + colorHex + '<br>' +
                                'H: ' + hue + '° | S: ' + sat + '% | L: ' + light + '%<br>' +
                                'Dominance: ' + perc.toFixed(1) + '%';
                            thumbnailDiv.style.display = 'block';
                        }
                    }
                });

                plot.on('plotly_unhover', function(data) {
                    thumbnailDiv.style.display = 'none';
                });
            }, 1000); // Wait 1 second for plot to initialize
        });
    </script>
    """

    # Get the HTML and inject our custom JavaScript
    html_str = fig.to_html(include_plotlyjs='cdn', full_html=False)
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>3D Color Space</title>
    <meta charset="utf-8">
</head>
<body>
{html_str}
{custom_js}
</body>
</html>"""

    return full_html


def create_plant_color_grid(metadata_file, masks_dir, output_dir, max_plants=None):
    """
    Create HTML grid showing each plant with thumbnail and color palette.
    Includes toggle between frequency and visual importance rankings.
    """
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    masks_path = Path(masks_dir)
    total = len(metadata) if max_plants is None else min(len(metadata), max_plants)

    html_parts = [
        '<!DOCTYPE html>',
        '<html><head>',
        '<meta charset="utf-8">',
        '<style>',
        'body { font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }',
        '.header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
        '.toggle-buttons { margin-top: 15px; }',
        '.toggle-btn { padding: 10px 20px; margin-right: 10px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }',
        '.toggle-btn.active { background: #2c3e50; }',
        '.toggle-btn:hover { background: #2980b9; }',
        '.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; }',
        '.plant-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
        '.plant-thumbnail { width: 100%; height: 200px; object-fit: contain; margin-bottom: 10px; background: #fafafa; border-radius: 4px; }',
        '.plant-name { font-size: 11px; margin-bottom: 10px; color: #333; font-weight: bold; word-break: break-word; }',
        '.palette { display: flex; gap: 5px; height: 50px; }',
        '.palette.hidden { display: none; }',
        '.color-swatch { flex: 1; border-radius: 4px; position: relative; }',
        '.color-info { font-size: 9px; color: white; text-shadow: 0 0 2px black; padding: 2px; }',
        '</style>',
        '</head><body>',
        '<div class="header">',
        '<h1>Plant Color Palettes</h1>',
        f'<p>Showing {total} plants</p>',
        '<div class="toggle-buttons">',
        '<button class="toggle-btn active" onclick="showRanking(\'frequency\')">Frequency</button>',
        '<button class="toggle-btn" onclick="showRanking(\'visual\')">Visual Importance</button>',
        '</div>',
        '</div>',
        '<div class="grid">'
    ]

    count = 0
    for mask_key, entry in metadata.items():
        if max_plants and count >= max_plants:
            break

        # Check for both ranking methods
        has_freq = 'colors_frequency' in entry and entry['colors_frequency']
        has_visual = 'colors_visual' in entry and entry['colors_visual']
        has_legacy = 'colors' in entry and entry['colors']

        if not (has_freq or has_legacy):
            continue

        plant_name = entry['image']
        colors_freq = entry.get('colors_frequency', entry.get('colors', []))
        colors_visual = entry.get('colors_visual', entry.get('colors', []))
        segmented_file = entry.get('segmented_file', '')
        segmented_path = masks_path / segmented_file if segmented_file else None

        html_parts.append('<div class="plant-card">')

        # Add thumbnail if available
        if segmented_path and segmented_path.exists():
            thumbnail = create_thumbnail_base64(segmented_path, max_size=300)
            if thumbnail:
                html_parts.append(f'<img src="{thumbnail}" class="plant-thumbnail" alt="{plant_name}">')

        html_parts.append(f'<div class="plant-name">{plant_name}</div>')

        # Frequency palette
        html_parts.append('<div class="palette palette-frequency">')
        for color in colors_freq:
            hex_color = color['hex']
            percentage = color['percentage']
            html_parts.append(
                f'<div class="color-swatch" style="background-color: {hex_color};" title="{hex_color}">'
                f'<div class="color-info">{percentage:.1f}%</div>'
                f'</div>'
            )
        html_parts.append('</div>')

        # Visual importance palette
        html_parts.append('<div class="palette palette-visual hidden">')
        for color in colors_visual:
            hex_color = color['hex']
            percentage = color['percentage']
            html_parts.append(
                f'<div class="color-swatch" style="background-color: {hex_color};" title="{hex_color}">'
                f'<div class="color-info">{percentage:.1f}%</div>'
                f'</div>'
            )
        html_parts.append('</div>')

        html_parts.append('</div>')
        count += 1

    html_parts.extend([
        '</div>',
        '<script>',
        'function showRanking(type) {',
        '  // Update button states',
        '  var buttons = document.querySelectorAll(".toggle-btn");',
        '  buttons.forEach(btn => btn.classList.remove("active"));',
        '  event.target.classList.add("active");',
        '  // Toggle palettes',
        '  var freqPalettes = document.querySelectorAll(".palette-frequency");',
        '  var visualPalettes = document.querySelectorAll(".palette-visual");',
        '  if (type === "frequency") {',
        '    freqPalettes.forEach(p => p.classList.remove("hidden"));',
        '    visualPalettes.forEach(p => p.classList.add("hidden"));',
        '  } else {',
        '    freqPalettes.forEach(p => p.classList.add("hidden"));',
        '    visualPalettes.forEach(p => p.classList.remove("hidden"));',
        '  }',
        '}',
        '</script>',
        '</body></html>'
    ])

    output_path = Path(output_dir) / 'plant_color_grid.html'
    with open(output_path, 'w') as f:
        f.write('\n'.join(html_parts))

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Visualize extracted plant colors'
    )
    parser.add_argument(
        '--masks-dir',
        default='masks',
        help='Directory containing masks metadata'
    )
    parser.add_argument(
        '--output-dir',
        default='visualizations',
        help='Directory to save visualizations'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1000,
        help='Sample size for 3D plot (for performance with large datasets)'
    )
    parser.add_argument(
        '--thumbnail-size',
        type=int,
        default=200,
        help='Thumbnail size in pixels'
    )

    args = parser.parse_args()

    # Setup
    metadata_file = Path(args.masks_dir) / 'masks_metadata.json'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading color data from {metadata_file}...")
    df_freq, df_visual = load_color_data(metadata_file, args.masks_dir)
    print(f"Loaded {len(df_freq)} frequency colors and {len(df_visual)} visual importance colors")
    print(f"From {df_freq['plant_id'].nunique()} plants")

    # Generate thumbnails (use frequency df as reference)
    df_freq = generate_thumbnails_for_plants(df_freq, max_size=args.thumbnail_size)
    df_visual = generate_thumbnails_for_plants(df_visual, max_size=args.thumbnail_size)

    # Create visualizations
    print("\nGenerating visualizations...")

    # 1. 3D color space (with thumbnails on hover and toggle)
    print("Creating 3D color space...")
    html_content = create_3d_color_space(df_freq, df_visual, sample_size=args.sample_size)
    with open(output_dir / 'color_space_3d.html', 'w') as f:
        f.write(html_content)
    print(f"✓ 3D color space: {output_dir / 'color_space_3d.html'}")

    # 2. Plant color grid
    print("Creating plant color grid...")
    grid_path = create_plant_color_grid(
        metadata_file,
        args.masks_dir,
        output_dir,
        max_plants=None  # Show all plants
    )
    print(f"✓ Plant color grid: {grid_path}")

    # Create index page
    num_plants = df_freq['plant_id'].nunique()
    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Flora Color Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 40px auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        .stats {{ background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-top: 30px; }}
        .viz-card {{ background: white; padding: 25px; border: 1px solid #ddd; border-radius: 8px; }}
        .viz-card h3 {{ margin-top: 0; color: #34495e; font-size: 1.3em; }}
        .viz-card p {{ color: #666; line-height: 1.6; }}
        .viz-card a {{ display: inline-block; margin-top: 15px; padding: 12px 24px; background: #3498db; color: white; text-decoration: none; border-radius: 4px; }}
        .viz-card a:hover {{ background: #2980b9; }}
        .info-box {{ background: #e8f4f8; padding: 15px; border-left: 4px solid #3498db; border-radius: 4px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>🌿 Flora Color Analysis</h1>

    <div class="stats">
        <h2>Dataset Summary</h2>
        <p><strong>Total plants analyzed:</strong> {num_plants}</p>
        <p><strong>Total color entries:</strong> {len(df_freq)} (per ranking method)</p>
        <p><strong>Average colors per plant:</strong> {len(df_freq) / num_plants:.1f}</p>
    </div>

    <div class="info-box">
        <p><strong>Two Ranking Methods:</strong></p>
        <p><strong>Frequency:</strong> Colors ranked by how much area they cover (most common colors first)</p>
        <p><strong>Visual Importance:</strong> Colors ranked by combined frequency (40%), saturation (30%), and contrast (30%) - highlights vivid, distinctive colors even if small</p>
    </div>

    <h2>Visualizations</h2>
    <div class="viz-grid">
        <div class="viz-card">
            <h3>🔮 3D Color Space Explorer</h3>
            <p>Interactive 3D visualization showing plant colors in HSL color space. Points naturally cluster by color similarity. Hover over any point to see a thumbnail preview of the plant.</p>
            <p><strong>Tips:</strong> Click and drag to rotate, scroll to zoom, hover to see plant details.</p>
            <a href="color_space_3d.html">Open 3D Explorer</a>
        </div>

        <div class="viz-card">
            <h3>🎴 Plant Color Grid</h3>
            <p>Grid view showing all plants with their segmented thumbnails and dominant color palettes. Great for browsing and comparing color patterns across specimens.</p>
            <p><strong>Showing:</strong> {num_plants} plants with thumbnails and color swatches.</p>
            <a href="plant_color_grid.html">Open Color Grid</a>
        </div>
    </div>

    <div style="margin-top: 40px; padding: 20px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px;">
        <p style="margin: 0;"><strong>Note:</strong> For large datasets (1000+ plants), the 3D color space uses sampling (showing {args.sample_size} points) for performance. Adjust with <code>--sample-size</code> parameter.</p>
    </div>
</body>
</html>"""

    index_path = output_dir / 'index.html'
    with open(index_path, 'w') as f:
        f.write(index_html)

    print(f"\n{'='*60}")
    print(f"✓ Visualizations complete!")
    print(f"{'='*60}")
    print(f"\n📊 Generated:")
    print(f"  • 3D Color Space (with thumbnail hover)")
    print(f"  • Plant Color Grid (with thumbnails)")
    print(f"\n🌐 Open in browser:")
    print(f"  {index_path.absolute()}")
    print()


if __name__ == '__main__':
    main()
