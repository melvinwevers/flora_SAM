#!/usr/bin/env python3
"""
Plant Embedding and Clustering Tool

Extracts embeddings from botanical illustrations using DINOv2,
then creates UMAP visualizations and clusters using HDBSCAN.
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import umap
from PIL import Image
from tqdm import tqdm
import transformers
import hdbscan

warnings.filterwarnings('ignore')


class EmbeddingExtractor:
    """Extract DINOv2 embeddings from images with caching."""

    def __init__(self, cache_dir: str = 'embeddings_cache', device: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")

        self.model = None
        self.processor = None

    def _load_dinov2(self):
        """Load DINOv2 base model."""
        if self.model is None:
            print("Loading DINOv2 base model...")
            self.processor = transformers.AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.model = transformers.AutoModel.from_pretrained('facebook/dinov2-base')
            self.model.to(self.device)
            self.model.eval()

    def _get_cache_path(self) -> Path:
        """Get cache file path for embeddings."""
        return self.cache_dir / "dinov2_embeddings.npz"

    def _load_cached_embeddings(self) -> Optional[Tuple[List[str], np.ndarray]]:
        """Load cached embeddings if they exist."""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            print("Loading cached DINOv2 embeddings...")
            data = np.load(cache_path, allow_pickle=True)
            return list(data['filenames']), data['embeddings']
        return None

    def _save_embeddings(self, filenames: List[str], embeddings: np.ndarray):
        """Save embeddings to cache."""
        cache_path = self._get_cache_path()
        np.savez_compressed(cache_path, filenames=filenames, embeddings=embeddings)
        print(f"Saved embeddings to {cache_path}")

    def _load_image(self, image_path: str) -> Image.Image:
        """Load and prepare image."""
        img = Image.open(image_path).convert('RGB')
        return img

    def _extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Extract DINOv2 embeddings for a batch of images."""
        if self.processor is None or self.model is None:
            raise RuntimeError("Model not loaded. Call _load_dinov2() first.")

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()

        return embeddings

    def extract_embeddings(
        self,
        image_paths: List[str],
        batch_size: int = 32,
        use_cache: bool = True
    ) -> Tuple[List[str], np.ndarray]:
        """Extract DINOv2 embeddings for all images."""

        # Try loading from cache
        if use_cache:
            cached = self._load_cached_embeddings()
            if cached is not None:
                return cached

        # Load model
        self._load_dinov2()

        # Extract embeddings in batches
        all_embeddings = []
        valid_paths = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting DINOv2 embeddings"):
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
                    embeddings = self._extract_batch(batch_images)
                    all_embeddings.append(embeddings)
                    valid_paths.extend(batch_valid_paths)
                except Exception as e:
                    print(f"Error extracting batch: {e}")
                    continue

        # Concatenate all embeddings
        embeddings_array = np.vstack(all_embeddings)

        # Save to cache
        self._save_embeddings(valid_paths, embeddings_array)

        return valid_paths, embeddings_array


class ClusterAnalyzer:
    """Compute UMAP projections and HDBSCAN clusters."""

    def __init__(self, cache_dir: str = 'embeddings_cache'):
        self.cache_dir = Path(cache_dir)
        self.umap_cache_path = self.cache_dir / 'umap_projection.npz'

    def compute_umap(
        self,
        embeddings: np.ndarray,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'cosine',
        random_state: int = 42,
        use_cache: bool = True
    ) -> np.ndarray:
        """Compute UMAP projection."""

        # Try loading from cache
        if use_cache and self.umap_cache_path.exists():
            print("Loading cached UMAP projection...")
            data = np.load(self.umap_cache_path)
            return data['projection']

        print("Computing UMAP projection...")
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_components=2,
            random_state=random_state
        )
        projection = reducer.fit_transform(embeddings)

        # Save to cache
        np.savez_compressed(self.umap_cache_path, projection=projection)
        print(f"Saved UMAP projection to {self.umap_cache_path}")

        return projection

    def compute_clusters(
        self,
        projection: np.ndarray,
        min_cluster_size: int = 15,
        min_samples: int = 5
    ) -> np.ndarray:
        """Compute HDBSCAN clusters for UMAP projection."""
        print("Clustering with HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean'
        )
        cluster_labels = clusterer.fit_predict(projection)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        print(f"Found {n_clusters} clusters, {n_noise} noise points")

        return cluster_labels

    def save_cluster_data(
        self,
        filenames: List[str],
        cluster_labels: np.ndarray,
        output_dir: str = 'visualizations/cluster_data'
    ):
        """Save cluster assignments in JSON format for prepare_data.py."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Group filenames by cluster
        cluster_dict = {}
        for filename, cluster_id in zip(filenames, cluster_labels):
            if cluster_id == -1:  # Skip noise
                continue

            cluster_id_str = str(cluster_id)
            if cluster_id_str not in cluster_dict:
                cluster_dict[cluster_id_str] = []

            # Extract plant_id from filename
            plant_id = Path(filename).stem.replace('.tif', '')
            cluster_dict[cluster_id_str].append(plant_id)

        # Save to JSON
        output_path = Path(output_dir) / 'cluster_data_dinov2.json'
        with open(output_path, 'w') as f:
            json.dump(cluster_dict, f, indent=2)

        print(f"Saved cluster data to {output_path}")
        print(f"Total clusters: {len(cluster_dict)}")
        for cluster_id, plants in sorted(cluster_dict.items(), key=lambda x: int(x[0])):
            print(f"  Cluster {cluster_id}: {len(plants)} plants")


def main():
    parser = argparse.ArgumentParser(description='Extract DINOv2 embeddings and create clusters')
    parser.add_argument('--input', type=str, default='masks/', help='Input directory with plant images')
    parser.add_argument('--output', type=str, default='visualizations/', help='Output directory')
    parser.add_argument('--cache-dir', type=str, default='embeddings_cache/', help='Cache directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for embedding extraction')
    parser.add_argument('--use-cache', action='store_true', help='Use cached embeddings/UMAP if available')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache before running')
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
    filenames, embeddings = extractor.extract_embeddings(
        image_paths,
        batch_size=args.batch_size,
        use_cache=args.use_cache
    )

    # Compute UMAP and clusters
    analyzer = ClusterAnalyzer(cache_dir=args.cache_dir)

    projection = analyzer.compute_umap(
        embeddings,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        use_cache=args.use_cache
    )

    cluster_labels = analyzer.compute_clusters(
        projection,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples
    )

    # Save cluster data for prepare_data.py
    analyzer.save_cluster_data(
        filenames,
        cluster_labels,
        output_dir=f"{args.output}/cluster_data"
    )

    # Save metadata
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    metadata = {
        'n_images': len(image_paths),
        'n_clusters': n_clusters,
        'n_noise': list(cluster_labels).count(-1),
        'model': 'dinov2-base',
        'umap_params': {
            'n_neighbors': args.n_neighbors,
            'min_dist': args.min_dist,
            'metric': 'cosine'
        },
        'hdbscan_params': {
            'min_cluster_size': args.min_cluster_size,
            'min_samples': args.min_samples
        }
    }

    metadata_path = Path(args.output) / 'cluster_analysis.json'
    metadata_path.parent.mkdir(exist_ok=True, parents=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")
    print(f"Cluster data: {args.output}/cluster_data/cluster_data_dinov2.json")
    print(f"Metadata: {metadata_path}")


if __name__ == '__main__':
    main()
