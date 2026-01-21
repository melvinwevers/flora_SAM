"""
Color Analysis page - analyze color patterns and relationships
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils import data_loader, charts

st.set_page_config(page_title="Color Analysis", page_icon="🎨", layout="wide")

st.title("🎨 Color Analysis")

st.markdown("""
Analyze color patterns across the collection and explore relationships between
colors, taxonomy, and visual clusters.
""")

# Load comparison metrics
try:
    metrics = data_loader.load_comparison_metrics()
except:
    metrics = {}

# Tabs for different analyses
tab1, tab2, tab3 = st.tabs([
    "Taxonomy vs Clusters",
    "Surprising Plants",
    "Statistics"
])

with tab1:
    st.header("Taxonomy vs Cluster Comparison")

    st.markdown("""
    Compare how taxonomic classification (family/genus) aligns with visual clustering.
    High agreement suggests visual similarity correlates with evolutionary relationships.
    """)

    # Model selection
    model_options = {
        'DINOv2': 'dinov2',
        'CLIP': 'clip',
        'PlantNet': 'plantnet',
        'Combined': 'combined'
    }

    selected_model_name = st.selectbox("Embedding Model", list(model_options.keys()))
    selected_model = model_options[selected_model_name]

    if selected_model in metrics:
        model_metrics = metrics[selected_model]

        # Display agreement scores
        st.subheader("Agreement Scores")

        col1, col2 = st.columns(2)

        ari = model_metrics.get('adjusted_rand_index', 0)
        nmi = model_metrics.get('normalized_mutual_info', 0)

        col1.metric(
            "Adjusted Rand Index",
            f"{ari:.3f}",
            help="Measures similarity between two clusterings (0=random, 1=perfect agreement)"
        )
        col2.metric(
            "Normalized Mutual Information",
            f"{nmi:.3f}",
            help="Measures mutual dependence between clusterings (0=independent, 1=perfect)"
        )

        # Contingency heatmap
        if 'contingency_matrix' in model_metrics:
            st.subheader("Family × Cluster Contingency Matrix")

            st.markdown("""
            Shows how many plants from each family fall into each cluster.
            Diagonal patterns suggest alignment between taxonomy and visual similarity.
            """)

            contingency = model_metrics['contingency_matrix']

            if contingency:
                fig = charts.create_contingency_heatmap(
                    contingency,
                    f"Family × Cluster ({selected_model_name})"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Purity scores
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cluster Purity")
            st.markdown("Percentage of dominant family in each cluster")

            if 'cluster_purity' in model_metrics:
                purity_data = model_metrics['cluster_purity']
                purity_df = pd.DataFrame(
                    list(purity_data.items()),
                    columns=['Cluster', 'Purity']
                )
                purity_df['Purity'] = purity_df['Purity'] * 100
                purity_df = purity_df.sort_values('Purity', ascending=False)

                st.dataframe(
                    purity_df.style.format({'Purity': '{:.1f}%'}),
                    use_container_width=True,
                    height=300
                )

        with col2:
            st.subheader("Family Concentration")
            st.markdown("Percentage of family in its dominant cluster")

            if 'family_concentration' in model_metrics:
                conc_data = model_metrics['family_concentration']
                conc_df = pd.DataFrame(
                    list(conc_data.items()),
                    columns=['Family', 'Concentration']
                )
                conc_df['Concentration'] = conc_df['Concentration'] * 100
                conc_df = conc_df.sort_values('Concentration', ascending=False)

                st.dataframe(
                    conc_df.style.format({'Concentration': '{:.1f}%'}),
                    use_container_width=True,
                    height=300
                )

    else:
        st.warning(f"No comparison metrics available for {selected_model_name}")

with tab2:
    st.header("Surprising Plants")

    st.markdown("""
    Plants that break expected patterns - visually similar but taxonomically distant,
    or vice versa. These are interesting edge cases for research.
    """)

    # Model selection
    selected_model_name2 = st.selectbox(
        "Embedding Model",
        list(model_options.keys()),
        key='surprising_model'
    )
    selected_model2 = model_options[selected_model_name2]

    if selected_model2 in metrics and 'surprising_plants' in metrics[selected_model2]:
        surprising = metrics[selected_model2]['surprising_plants']

        if surprising:
            st.subheader(f"Plants in minority families within their cluster ({selected_model_name2})")

            surprising_df = pd.DataFrame(surprising)
            st.dataframe(
                surprising_df,
                use_container_width=True,
                column_config={
                    'plant_id': 'Plant ID',
                    'family': 'Plant Family',
                    'cluster': 'Cluster',
                    'dominant_family': 'Cluster\'s Dominant Family',
                    'cluster_size': 'Cluster Size'
                }
            )

            # Display thumbnails of first few surprising plants
            st.subheader("Examples")

            cols = st.columns(5)

            for idx, (col, row) in enumerate(zip(cols, surprising_df.head(5).itertuples())):
                with col:
                    plant_id = row.plant_id
                    thumbnail_path = data_loader.get_thumbnail_path(plant_id)

                    if thumbnail_path.exists():
                        st.image(str(thumbnail_path), use_container_width=True)
                    st.caption(f"**{plant_id}**")
                    st.caption(f"Family: {row.family}")
                    st.caption(f"In cluster dominated by {row.dominant_family}")

        else:
            st.info(f"No surprising plants identified for {selected_model_name2}")
    else:
        st.warning(f"No surprising plants data available for {selected_model_name2}")

with tab3:
    st.header("Dataset Statistics")

    # Overall stats
    plants_df = data_loader.load_plants_metadata()

    st.subheader("Taxonomy Coverage")

    col1, col2, col3 = st.columns(3)

    total_plants = len(plants_df)
    with_family = plants_df['family'].notna().sum()
    with_genus = plants_df['genus'].notna().sum()

    col1.metric("Total Plants", f"{total_plants:,}")
    col2.metric("With Family", f"{with_family:,} ({with_family/total_plants*100:.1f}%)")
    col3.metric("With Genus", f"{with_genus:,} ({with_genus/total_plants*100:.1f}%)")

    # Family distribution
    st.subheader("Top Families by Size")

    family_counts = plants_df['family'].value_counts().head(15)

    fig = charts.create_bar_chart(
        family_counts,
        "Family",
        "Number of Plants",
        "Top 15 Families"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster sizes across models
    st.subheader("Cluster Counts by Model")

    clusters_df = data_loader.load_cluster_assignments()

    cluster_counts = {}
    for model in ['dinov2', 'clip', 'plantnet', 'combined']:
        if model in clusters_df.columns:
            n_clusters = clusters_df[clusters_df[model] >= 0][model].nunique()
            cluster_counts[model.upper()] = n_clusters

    if cluster_counts:
        fig = charts.create_bar_chart(
            cluster_counts,
            "Model",
            "Number of Clusters",
            "Clusters Detected by Each Model"
        )
        st.plotly_chart(fig, use_container_width=True)
