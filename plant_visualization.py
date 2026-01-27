#!/usr/bin/env python3
"""
Plant UMAP Visualization Tool

Extracts embeddings from botanical illustrations using DINOv2, CLIP, and PlantNet,
then creates interactive UMAP visualizations with outlier detection.
"""

import argparse
import io
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import umap
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import transformers
import timm
import hdbscan

from thumbnail_cache import ThumbnailCache

warnings.filterwarnings('ignore')


class EmbeddingExtractor:
    """Extract embeddings from images using multiple models with caching."""

    def __init__(self, cache_dir: str = 'embeddings_cache', device: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")

        self.models = {}
        self.processors = {}

    def _load_dinov2(self):
        """Load DINOv2 base model."""
        if 'dinov2' not in self.models:
            print("Loading DINOv2 base model...")
            processor = transformers.AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            model = transformers.AutoModel.from_pretrained('facebook/dinov2-base')
            model.to(self.device)
            model.eval()
            self.models['dinov2'] = model
            self.processors['dinov2'] = processor

    def _load_clip(self):
        """Load CLIP base model."""
        if 'clip' not in self.models:
            print("Loading CLIP base model...")
            processor = transformers.CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
            model = transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
            model.to(self.device)
            model.eval()
            self.models['clip'] = model
            self.processors['clip'] = processor

    def _load_plantnet(self):
        """Load PlantNet model (using timm)."""
        if 'plantnet' not in self.models:
            print("Loading PlantNet model...")
            # Using smaller ViT model for faster inference
            model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)

            data_config = timm.data.resolve_model_data_config(model)
            transforms = timm.data.create_transform(**data_config, is_training=False)

            model.to(self.device)
            model.eval()
            self.models['plantnet'] = model
            self.processors['plantnet'] = transforms

    def _get_cache_path(self, model_name: str) -> Path:
        """Get cache file path for a model."""
        return self.cache_dir / f"{model_name}_embeddings.npz"

    def _load_cached_embeddings(self, model_name: str) -> Optional[Tuple[List[str], np.ndarray]]:
        """Load cached embeddings if they exist."""
        cache_path = self._get_cache_path(model_name)
        if cache_path.exists():
            print(f"Loading cached {model_name} embeddings...")
            data = np.load(cache_path, allow_pickle=True)
            return list(data['filenames']), data['embeddings']
        return None

    def _save_embeddings(self, model_name: str, filenames: List[str], embeddings: np.ndarray):
        """Save embeddings to cache."""
        cache_path = self._get_cache_path(model_name)
        np.savez_compressed(cache_path, filenames=filenames, embeddings=embeddings)
        print(f"Saved {model_name} embeddings to cache")

    def _load_image(self, image_path: str) -> Image.Image:
        """Load and prepare image."""
        img = Image.open(image_path).convert('RGB')
        return img

    def _extract_dinov2_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Extract DINOv2 embeddings for a batch of images."""
        inputs = self.processors['dinov2'](images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.models['dinov2'](**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()

        return embeddings

    def _extract_clip_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Extract CLIP embeddings for a batch of images."""
        inputs = self.processors['clip'](images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.models['clip'].get_image_features(**inputs)
            embeddings = image_features.cpu().numpy()

        return embeddings

    def _extract_plantnet_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Extract PlantNet embeddings for a batch of images."""
        # Apply transforms
        tensors = torch.stack([self.processors['plantnet'](img) for img in images])
        tensors = tensors.to(self.device)

        with torch.no_grad():
            embeddings = self.models['plantnet'](tensors).cpu().numpy()

        return embeddings

    def extract_embeddings(
        self,
        image_paths: List[str],
        model_name: str,
        batch_size: int = 32,
        use_cache: bool = True
    ) -> Tuple[List[str], np.ndarray]:
        """Extract embeddings for all images using specified model."""

        # Try loading from cache
        if use_cache:
            cached = self._load_cached_embeddings(model_name)
            if cached is not None:
                return cached

        # Load model
        if model_name == 'dinov2':
            self._load_dinov2()
            extract_fn = self._extract_dinov2_batch
        elif model_name == 'clip':
            self._load_clip()
            extract_fn = self._extract_clip_batch
        elif model_name == 'plantnet':
            self._load_plantnet()
            extract_fn = self._extract_plantnet_batch
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Extract embeddings in batches
        all_embeddings = []
        valid_paths = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Extracting {model_name}"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            batch_valid_paths = []

            for path in batch_paths:
                try:
                    img = self._load_image(path)
                    batch_images.append(img)
                    batch_valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue

            if batch_images:
                try:
                    embeddings = extract_fn(batch_images)
                    all_embeddings.append(embeddings)
                    valid_paths.extend(batch_valid_paths)
                except Exception as e:
                    print(f"Error extracting batch: {e}")
                    continue

        # Concatenate all embeddings
        embeddings_array = np.vstack(all_embeddings)

        # Save to cache
        self._save_embeddings(model_name, valid_paths, embeddings_array)

        return valid_paths, embeddings_array

    def extract_all(
        self,
        image_paths: List[str],
        batch_size: int = 32,
        use_cache: bool = True
    ) -> Dict[str, Tuple[List[str], np.ndarray]]:
        """Extract embeddings using all three models."""
        results = {}

        for model_name in ['dinov2', 'clip', 'plantnet']:
            filenames, embeddings = self.extract_embeddings(
                image_paths, model_name, batch_size, use_cache
            )
            results[model_name] = (filenames, embeddings)

        return results


class UMAPVisualizer:
    """Compute UMAP projections and create visualizations."""

    def __init__(self, cache_dir: str = 'embeddings_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_path = self.cache_dir / 'umap_projections.npz'

    def compute_umap(
        self,
        embeddings: np.ndarray,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'cosine',
        random_state: int = 42
    ) -> np.ndarray:
        """Compute UMAP projection."""
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_components=2,
            random_state=random_state
        )
        projection = reducer.fit_transform(embeddings)
        return projection

    def compute_all_projections(
        self,
        embeddings_dict: Dict[str, Tuple[List[str], np.ndarray]],
        use_cache: bool = True,
        **umap_params
    ) -> Dict[str, np.ndarray]:
        """Compute UMAP projections for all embeddings."""

        # Try loading from cache
        if use_cache and self.cache_path.exists():
            print("Loading cached UMAP projections...")
            data = np.load(self.cache_path, allow_pickle=True)
            return {k: data[k] for k in data.files}

        projections = {}

        # Individual embeddings
        for name, (_, embeddings) in embeddings_dict.items():
            print(f"Computing UMAP for {name}...")
            projections[name] = self.compute_umap(embeddings, **umap_params)

        # Combined embeddings
        print("Computing UMAP for combined embeddings...")
        combined = np.hstack([emb for _, emb in embeddings_dict.values()])
        projections['combined'] = self.compute_umap(combined, **umap_params)

        # Save to cache
        np.savez_compressed(self.cache_path, **projections)
        print("Saved UMAP projections to cache")

        return projections

    def compute_clusters(
        self,
        projections: Dict[str, np.ndarray],
        min_cluster_size: int = 15,
        min_samples: int = 5
    ) -> Dict[str, np.ndarray]:
        """Compute HDBSCAN clusters for each UMAP projection."""
        clusters = {}

        for name, projection in projections.items():
            print(f"Clustering {name} projection...")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )
            cluster_labels = clusterer.fit_predict(projection)
            clusters[name] = cluster_labels
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            print(f"  Found {n_clusters} clusters, {n_noise} noise points")

        return clusters

    def create_cluster_grid(
        self,
        filenames: List[str],
        projections: Dict[str, np.ndarray],
        clusters: Dict[str, np.ndarray],
        thumbnail_cache: ThumbnailCache,
        output_dir: str = 'visualizations/clusters',
        max_clusters: int = 50
    ):
        """Create grid visualizations showing sample images from each cluster."""
        from collections import defaultdict

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # For each embedding type, create cluster grids
        for name in projections.keys():
            print(f"Creating cluster grid for {name}...")
            cluster_labels = clusters[name]

            # Group images by cluster
            cluster_images = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Skip noise
                    cluster_images[label].append(filenames[i])

            # Limit to largest clusters
            sorted_clusters = sorted(cluster_images.items(), key=lambda x: len(x[1]), reverse=True)
            sorted_clusters = sorted_clusters[:max_clusters]

            # Create HTML grid for this embedding
            html = ["<html><head><title>Cluster Grid - {}</title>".format(name)]
            html.append("<style>")
            html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
            html.append("h1, h2 { color: #333; }")
            html.append(".cluster { margin: 30px 0; padding: 20px; background: #f9f9f9; border-radius: 8px; }")
            html.append(".grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }")
            html.append(".grid img { width: 100%; height: auto; border: 1px solid #ddd; }")
            html.append("</style></head><body>")

            html.append(f"<h1>Cluster Grid - {name}</h1>")
            html.append(f"<p>Showing {len(sorted_clusters)} largest clusters (of {len(cluster_images)} total)</p>")

            # Show each cluster
            for cluster_id, images_list in sorted_clusters:
                images = images_list[:12]  # Max 12 images per cluster

                html.append(f"<div class='cluster'>")
                html.append(f"<h2>Cluster {cluster_id} ({len(images_list)} images)</h2>")
                html.append("<div class='grid'>")

                for img_path in images:
                    thumbnail = thumbnail_cache.get(img_path, max_size=120)
                    if thumbnail:
                        html.append(f'<img src="{thumbnail}" title="{Path(img_path).name}">')
                    else:
                        print(f"Error loading {img_path}")

                html.append("</div></div>")

            html.append("</body></html>")

            # Save cluster grid
            cluster_file = output_dir / f"cluster_grid_{name}.html"
            with open(cluster_file, 'w') as f:
                f.write('\n'.join(html))

            print(f"Saved cluster grid to {cluster_file}")

    def create_interactive_dashboard(
        self,
        projections: Dict[str, np.ndarray],
        embeddings_dict: Dict[str, Tuple[List[str], np.ndarray]],
        clusters: Dict[str, np.ndarray],
        thumbnail_cache: ThumbnailCache,
        output_path: str = 'visualizations/umap_interactive.html',
        max_thumbnails: int = 200
    ):
        """Create interactive Plotly dashboard with clusters and thumbnail preview."""
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        # Get common filenames (from first embedding)
        filenames = list(embeddings_dict.values())[0][0]

        # Pre-generate thumbnails for sampling
        print("Generating thumbnails for preview...")
        thumbnail_index_cache = {}

        # Sample images for thumbnail generation
        if len(filenames) > max_thumbnails:
            sample_indices = np.random.choice(len(filenames), max_thumbnails, replace=False)
            sample_indices = set(sample_indices)
        else:
            sample_indices = set(range(len(filenames)))

        for i in sample_indices:
            thumbnail = thumbnail_cache.get(filenames[i], max_size=200)
            if thumbnail:
                # Extract just the base64 part (remove data:image/png;base64, prefix)
                img_str = thumbnail.split(',', 1)[1] if ',' in thumbnail else thumbnail
                thumbnail_index_cache[i] = img_str

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['DINOv2', 'CLIP', 'PlantNet', 'Combined'],
            horizontal_spacing=0.1,
            vertical_spacing=0.1
        )

        # Add scatter plots
        subplot_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for (name, projection), (row, col) in zip(projections.items(), subplot_positions):
            cluster_labels = clusters[name]
            hover_text = []
            customdata = []

            for i, fname in enumerate(filenames):
                cluster_id = int(cluster_labels[i])
                cluster_str = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"

                # Create hover text
                hover_parts = [
                    f"File: {Path(fname).name}",
                    f"{cluster_str}",
                    f"X: {projection[i, 0]:.2f}, Y: {projection[i, 1]:.2f}"
                ]

                # Add thumbnail to hover if available
                if i in thumbnail_index_cache:
                    img_str = thumbnail_index_cache[i]
                    hover_parts.insert(0, f'<img src="data:image/png;base64,{img_str}">')

                hover_text.append('<br>'.join(hover_parts))
                customdata.append(fname)

            fig.add_trace(
                go.Scattergl(
                    x=projection[:, 0],
                    y=projection[:, 1],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=cluster_labels,
                        colorscale='Viridis',
                        opacity=0.7,
                        line=dict(width=0),
                        showscale=True
                    ),
                    hovertext=hover_text,
                    hoverinfo='text',
                    customdata=customdata,
                    name=name
                ),
                row=row, col=col
            )

        # Update layout
        fig.update_layout(
            title='Plant UMAP Visualization with HDBSCAN Clustering (hover to see thumbnails)',
            height=1000,
            showlegend=False,
            hovermode='closest'
        )

        fig.write_html(str(output_path))
        print(f"Saved interactive dashboard to {output_path}")


class OutlierDetector:
    """Detect outliers using multiple methods."""

    def __init__(self):
        pass

    def detect_lof_outliers(self, umap_coords: np.ndarray, contamination: float = 0.05) -> np.ndarray:
        """Detect outliers using Local Outlier Factor."""
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        predictions = lof.fit_predict(umap_coords)
        return predictions == -1

    def detect_distance_outliers(self, umap_coords: np.ndarray, percentile: float = 95) -> np.ndarray:
        """Detect outliers based on distance to nearest neighbors."""
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=6).fit(umap_coords)
        distances, _ = nbrs.kneighbors(umap_coords)
        # Use distance to 5th nearest neighbor
        dist_to_5th = distances[:, 5]
        threshold = np.percentile(dist_to_5th, percentile)
        return dist_to_5th > threshold

    def detect_embedding_disagreement(
        self,
        embeddings_dict: Dict[str, Tuple[List[str], np.ndarray]],
        n_clusters: int = 30
    ) -> Tuple[np.ndarray, Dict]:
        """Detect outliers based on cluster disagreement across embeddings."""
        cluster_assignments = {}

        for name, (_, embeddings) in embeddings_dict.items():
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings)
            cluster_assignments[name] = clusters

        # Count disagreements
        n_samples = len(list(cluster_assignments.values())[0])
        disagreement_scores = np.zeros(n_samples)

        names = list(cluster_assignments.keys())
        for i in range(n_samples):
            # Count unique cluster assignments across models
            clusters_for_sample = [cluster_assignments[name][i] for name in names]
            unique_clusters = len(set(clusters_for_sample))
            disagreement_scores[i] = unique_clusters

        # Flag if assigned to different clusters in all 3 models
        is_outlier = disagreement_scores == 3

        return is_outlier, cluster_assignments

    def detect_small_clusters(
        self,
        cluster_assignments: Dict,
        min_size: int = 3
    ) -> np.ndarray:
        """Detect points in very small clusters."""
        # Use first clustering
        clusters = list(cluster_assignments.values())[0]

        # Count cluster sizes
        unique, counts = np.unique(clusters, return_counts=True)
        small_clusters = unique[counts < min_size]

        # Mark points in small clusters
        is_outlier = np.isin(clusters, small_clusters)

        return is_outlier

    def detect(
        self,
        projections: Dict[str, np.ndarray],
        embeddings_dict: Dict[str, Tuple[List[str], np.ndarray]]
    ) -> Dict:
        """Run all outlier detection methods and combine results."""
        filenames = list(embeddings_dict.values())[0][0]
        n_samples = len(filenames)

        # Use combined UMAP for spatial outliers
        umap_coords = projections['combined']

        print("Detecting outliers...")

        # Method 1: LOF
        lof_outliers = self.detect_lof_outliers(umap_coords)

        # Method 2: Distance-based
        dist_outliers = self.detect_distance_outliers(umap_coords)

        # Method 3: Embedding disagreement
        disagree_outliers, cluster_assignments = self.detect_embedding_disagreement(embeddings_dict)

        # Method 4: Small clusters
        small_cluster_outliers = self.detect_small_clusters(cluster_assignments)

        # Combine methods - only flag if first 3 methods all agree
        # (LOF, Distance, Disagreement - excluding SmallCluster)
        outlier_votes = np.zeros(n_samples)
        outlier_votes += lof_outliers.astype(int)
        outlier_votes += dist_outliers.astype(int)
        outlier_votes += disagree_outliers.astype(int)

        # Only report if all 3 core methods flag it
        high_confidence = [filenames[i] for i in range(n_samples) if outlier_votes[i] == 3]
        medium_confidence = []
        low_confidence = []

        # Build method details
        method_details = {}
        method_names = ['LOF', 'Distance', 'Disagreement', 'SmallCluster']
        method_flags = [lof_outliers, dist_outliers, disagree_outliers, small_cluster_outliers]

        for i, fname in enumerate(filenames):
            if outlier_votes[i] > 0:
                methods = [name for name, flags in zip(method_names, method_flags) if flags[i]]
                method_details[fname] = methods

        print(f"Found {len(high_confidence)} high-confidence outliers")
        print(f"Found {len(medium_confidence)} medium-confidence outliers")
        print(f"Found {len(low_confidence)} low-confidence outliers")

        return {
            'high_confidence': high_confidence,
            'medium_confidence': medium_confidence,
            'low_confidence': low_confidence,
            'method_details': method_details
        }

    def create_outlier_report(
        self,
        outliers: Dict,
        thumbnail_cache: ThumbnailCache,
        output_path: str = 'visualizations/outliers_report.html'
    ):
        """Create HTML report of outliers with diagnostic information."""
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        html = ["<html><head><title>Outlier Report</title>"]
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("h1, h2 { color: #333; }")
        html.append(".info { background: #f0f0f0; padding: 15px; margin: 20px 0; border-radius: 5px; }")
        html.append(".outlier-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; }")
        html.append(".outlier-item { border: 1px solid #ddd; padding: 10px; }")
        html.append(".outlier-item img { max-width: 100%; height: auto; }")
        html.append(".methods { font-size: 0.9em; color: #666; margin-top: 5px; }")
        html.append(".explanation { font-size: 0.85em; color: #444; margin-top: 10px; padding: 8px; background: #fff3cd; border-radius: 3px; }")
        html.append("</style></head><body>")

        html.append("<h1>Plant Segmentation Outlier Report</h1>")

        total = len(outliers['high_confidence']) + len(outliers['medium_confidence']) + len(outliers['low_confidence'])

        html.append("<div class='info'>")
        html.append(f"<p><strong>Total outliers detected:</strong> {total} images (out of all processed)</p>")
        html.append("<p><strong>Detection criteria:</strong> Images flagged by all 3 methods:</p>")
        html.append("<ul>")
        html.append("<li><strong>LOF:</strong> Point is isolated in UMAP space (low local density)</li>")
        html.append("<li><strong>Distance:</strong> Point is far from its nearest neighbors (top 5% most distant)</li>")
        html.append("<li><strong>Disagreement:</strong> DINOv2, CLIP, and PlantNet assign it to different clusters</li>")
        html.append("</ul>")
        html.append("<p><strong>Common reasons for flagging:</strong></p>")
        html.append("<ul>")
        html.append("<li>Poor segmentation (too much background, multiple objects, cropping errors)</li>")
        html.append("<li>Unusual plant specimens (very different from typical plants in dataset)</li>")
        html.append("<li>Low quality images (blurry, faded, damaged illustrations)</li>")
        html.append("<li>Non-plant content (text, diagrams, annotations mistaken as plants)</li>")
        html.append("</ul>")
        html.append("</div>")

        if total == 0:
            html.append("<p>No outliers detected! All images passed the quality checks.</p>")

        for confidence_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
            level_name = confidence_level.replace('_', ' ').title()
            items = outliers[confidence_level]

            if not items:
                continue

            html.append(f"<h2>{level_name} ({len(items)} items)</h2>")
            html.append("<div class='outlier-grid'>")

            for fname in items[:50]:  # Limit to 50 per category
                methods = ', '.join(outliers['method_details'].get(fname, []))

                # Embed image as base64 using cache
                thumbnail = thumbnail_cache.get(fname, max_size=300)
                if thumbnail:
                    img_tag = f'<img src="{thumbnail}" alt="{Path(fname).name}">'
                else:
                    img_tag = f'<p>Error loading image</p>'

                html.append("<div class='outlier-item'>")
                html.append(img_tag)
                html.append(f"<p><strong>{Path(fname).name}</strong></p>")
                html.append(f"<p class='methods'>Flagged by: {methods}</p>")
                html.append("<div class='explanation'>")
                html.append("<strong>What to look for:</strong><br>")
                html.append("• Is the segmentation clean or does it include background/borders?<br>")
                html.append("• Is this a typical plant or something unusual/non-botanical?<br>")
                html.append("• Is the image quality good or degraded?")
                html.append("</div>")
                html.append("</div>")

            html.append("</div>")

        html.append("</body></html>")

        with open(output_path, 'w') as f:
            f.write('\n'.join(html))

        print(f"Saved outlier report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plant UMAP Visualization')
    parser.add_argument('--input', type=str, default='masks/', help='Input directory with plant images')
    parser.add_argument('--output', type=str, default='visualizations/', help='Output directory')
    parser.add_argument('--cache-dir', type=str, default='embeddings_cache/', help='Cache directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for embedding extraction')
    parser.add_argument('--use-cache', action='store_true', help='Use cached embeddings if available')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache before running')
    parser.add_argument('--recompute-umap', action='store_true', help='Recompute UMAP projections')
    parser.add_argument('--n-neighbors', type=int, default=15, help='UMAP n_neighbors parameter')
    parser.add_argument('--min-dist', type=float, default=0.1, help='UMAP min_dist parameter')
    parser.add_argument('--min-cluster-size', type=int, default=15, help='HDBSCAN min_cluster_size parameter')
    parser.add_argument('--min-samples', type=int, default=5, help='HDBSCAN min_samples parameter')

    args = parser.parse_args()

    # Get image paths
    input_dir = Path(args.input)
    image_paths = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))
    image_paths = [str(p) for p in image_paths]

    print(f"Found {len(image_paths)} images")

    if not image_paths:
        print("No images found!")
        return

    # Clear cache if requested
    if args.clear_cache:
        cache_dir = Path(args.cache_dir)
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
        print("Cache cleared")

    # Extract embeddings
    extractor = EmbeddingExtractor(cache_dir=args.cache_dir)
    embeddings_dict = extractor.extract_all(
        image_paths,
        batch_size=args.batch_size,
        use_cache=args.use_cache
    )

    # Compute UMAP
    visualizer = UMAPVisualizer(cache_dir=args.cache_dir)
    use_umap_cache = args.use_cache and not args.recompute_umap

    projections = visualizer.compute_all_projections(
        embeddings_dict,
        use_cache=use_umap_cache,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )

    # Compute clusters
    clusters = visualizer.compute_clusters(projections)

    # Create shared thumbnail cache
    print("Initializing thumbnail cache...")
    thumbnail_cache = ThumbnailCache()

    # Create cluster grids
    visualizer.create_cluster_grid(
        list(embeddings_dict.values())[0][0],
        projections,
        clusters,
        thumbnail_cache,
        output_dir=f"{args.output}/clusters"
    )

    # Create interactive dashboard
    visualizer.create_interactive_dashboard(
        projections,
        embeddings_dict,
        clusters,
        thumbnail_cache,
        output_path=f"{args.output}/umap_interactive.html"
    )

    # Save cluster assignments
    filenames = list(embeddings_dict.values())[0][0]
    cluster_assignments_path = Path(args.cache_dir) / 'cluster_assignments.npz'

    # Convert cluster labels to numpy arrays and save with filenames
    cluster_data = {name: labels for name, labels in clusters.items()}
    np.savez_compressed(
        cluster_assignments_path,
        filenames=np.array(filenames, dtype=object),
        **cluster_data
    )
    print(f"Saved cluster assignments to {cluster_assignments_path}")

    # Save metadata
    n_clusters_per_embedding = {
        name: len(set(labels)) - (1 if -1 in labels else 0)
        for name, labels in clusters.items()
    }

    metadata = {
        'n_images': len(image_paths),
        'n_clusters': n_clusters_per_embedding,
        'umap_params': {
            'n_neighbors': args.n_neighbors,
            'min_dist': args.min_dist
        },
        'hdbscan_params': {
            'min_cluster_size': args.min_cluster_size,
            'min_samples': args.min_samples
        }
    }

    with open(f"{args.output}/cluster_analysis.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")
    print(f"Interactive dashboard: {args.output}/umap_interactive.html")
    print(f"Cluster grids: {args.output}/clusters/")


if __name__ == '__main__':
    main()
