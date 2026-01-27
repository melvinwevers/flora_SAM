"""
Reusable chart components using Plotly.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_thumbnail_grid(df, thumbnails_dir, columns=4):
    """
    Create HTML for a grid of plant thumbnails.

    Args:
        df: DataFrame with plant_id and thumbnail info
        thumbnails_dir: Path to thumbnails directory
        columns: Number of columns in grid

    Returns:
        HTML string for the grid
    """
    html_parts = ['<div style="display: grid; grid-template-columns: repeat({}, 1fr); gap: 10px;">'.format(columns)]

    for _, row in df.iterrows():
        plant_id = row['plant_id']
        thumb_path = f"{thumbnails_dir}/{plant_id}.jpg"

        # Create card for each plant
        card_html = f'''
        <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; text-align: center;">
            <img src="{thumb_path}" style="width: 100%; height: auto; border-radius: 3px;" />
            <p style="margin-top: 5px; font-size: 12px; color: #666;">{plant_id}</p>
        </div>
        '''
        html_parts.append(card_html)

    html_parts.append('</div>')
    return ''.join(html_parts)

def create_color_palette_display(colors):
    """
    Create HTML for displaying a color palette.

    Args:
        colors: List of dicts with 'hex' and 'percentage' keys

    Returns:
        HTML string for the palette
    """
    if not colors:
        return ""

    swatches = []
    for color in colors:
        hex_code = color.get('hex', '')
        percentage = color.get('percentage', 0)

        # Skip invalid entries
        if not hex_code or str(hex_code) == 'nan':
            continue

        try:
            pct_str = f"{float(percentage):.1f}%"
        except (ValueError, TypeError):
            pct_str = ""

        swatches.append(
            f'<div style="text-align:center;flex:0 0 auto;">'
            f'<div style="width:32px;height:32px;background:{hex_code};border:1px solid #ccc;border-radius:4px;"></div>'
            f'<div style="font-size:9px;margin-top:3px;line-height:1;white-space:nowrap;">{pct_str}</div>'
            f'</div>'
        )

    if not swatches:
        return ""

    return f'<div style="display:flex;align-items:flex-start;flex-wrap:wrap;gap:2px;max-width:100%;">{"".join(swatches)}</div>'

def create_color_palette_circles(colors):
    """
    Create HTML for displaying a color palette as circles.

    Args:
        colors: List of dicts with 'hex' and 'percentage' keys

    Returns:
        HTML string for the palette
    """
    if not colors:
        return ""

    swatches = []
    for color in colors:
        hex_code = color.get('hex', '')
        percentage = color.get('percentage', 0)

        # Skip invalid entries
        if not hex_code or str(hex_code) == 'nan':
            continue

        try:
            pct_str = f"{float(percentage):.1f}%"
        except (ValueError, TypeError):
            pct_str = ""

        swatches.append(
            f'<div style="text-align:center;flex:0 0 auto;">'
            f'<div style="width:32px;height:32px;background:{hex_code};border:1px solid #ccc;border-radius:50%;"></div>'
            f'<div style="font-size:9px;margin-top:3px;line-height:1;white-space:nowrap;">{pct_str}</div>'
            f'</div>'
        )

    if not swatches:
        return ""

    return f'<div style="display:flex;align-items:flex-start;flex-wrap:wrap;gap:2px;max-width:100%;">{"".join(swatches)}</div>'

def create_color_palette_bar(colors):
    """
    Create HTML for displaying a color palette as a horizontal bar.

    Args:
        colors: List of dicts with 'hex' and 'percentage' keys

    Returns:
        HTML string for the palette
    """
    if not colors:
        return ""

    # Calculate total percentage for proportional sizing
    total_pct = sum(float(c.get('percentage', 0)) for c in colors if c.get('hex') and str(c.get('hex')) != 'nan')

    if total_pct == 0:
        return ""

    segments = []
    for color in colors:
        hex_code = color.get('hex', '')
        percentage = color.get('percentage', 0)

        # Skip invalid entries
        if not hex_code or str(hex_code) == 'nan':
            continue

        try:
            pct = float(percentage)
            width_pct = (pct / total_pct) * 100 if total_pct > 0 else 0
            segments.append(
                f'<div style="flex:{width_pct};background:{hex_code};height:20px;" title="{pct:.1f}%"></div>'
            )
        except (ValueError, TypeError):
            continue

    if not segments:
        return ""

    return f'<div style="display:flex;width:100%;border:1px solid #ccc;border-radius:4px;overflow:hidden;">{"".join(segments)}</div>'

def create_cluster_distribution_pie(df, cluster_column, title="Cluster Distribution"):
    """
    Create pie chart showing distribution across clusters.

    Args:
        df: DataFrame with cluster assignments
        cluster_column: Name of cluster column
        title: Chart title

    Returns:
        Plotly figure
    """
    # Count plants per cluster
    cluster_counts = df[df[cluster_column] >= 0][cluster_column].value_counts().sort_index()

    fig = px.pie(
        values=cluster_counts.values,
        names=[f"Cluster {i}" for i in cluster_counts.index],
        title=title
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True, height=400)

    return fig

def create_contingency_heatmap(contingency_matrix, title="Family × Cluster Contingency"):
    """
    Create heatmap of contingency matrix.

    Args:
        contingency_matrix: Dict of family -> cluster -> count
        title: Chart title

    Returns:
        Plotly figure
    """
    # Convert dict to DataFrame
    df = pd.DataFrame(contingency_matrix).fillna(0)

    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=[f"Cluster {c}" for c in df.columns],
        y=df.index,
        colorscale='Blues',
        text=df.values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='Family: %{y}<br>Cluster: %{x}<br>Count: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Cluster",
        yaxis_title="Family",
        height=max(400, len(df) * 20),
        width=800
    )

    return fig

def create_scatter_plot(df, x_col, y_col, color_col=None, hover_data=None, title="Scatter Plot"):
    """
    Create scatter plot for color features.

    Args:
        df: DataFrame with data
        x_col: Column for x-axis
        y_col: Column for y-axis
        color_col: Optional column for coloring points
        hover_data: Optional list of columns to show on hover
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_data=hover_data or ['plant_id'],
        title=title,
        opacity=0.6
    )

    fig.update_traces(marker=dict(size=8))
    fig.update_layout(height=600, width=800)

    return fig

def create_umap_plot(df, x_col, y_col, color_col, title="UMAP Projection"):
    """
    Create UMAP scatter plot.

    Args:
        df: DataFrame with UMAP coordinates
        x_col: Column for x-axis (UMAP 1)
        y_col: Column for y-axis (UMAP 2)
        color_col: Column for coloring (cluster or family)
        title: Chart title

    Returns:
        Plotly figure
    """
    # Handle both numeric and categorical color columns
    if pd.api.types.is_numeric_dtype(df[color_col]):
        # Numeric: treat as categorical for clusters
        df[color_col] = df[color_col].astype(str)

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_data=['plant_id'],
        title=title,
        opacity=0.7
    )

    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        height=600,
        width=800,
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        showlegend=True
    )

    return fig

def create_bar_chart(data, x_label, y_label, title="Bar Chart"):
    """
    Create bar chart from data.

    Args:
        data: Dict or Series with categories and values
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Chart title

    Returns:
        Plotly figure
    """
    if isinstance(data, dict):
        df = pd.DataFrame(list(data.items()), columns=[x_label, y_label])
    else:
        df = pd.DataFrame({x_label: data.index, y_label: data.values})

    fig = px.bar(
        df,
        x=x_label,
        y=y_label,
        title=title
    )

    fig.update_layout(height=400, width=600)

    return fig
