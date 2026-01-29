#!/usr/bin/env python3
"""
Shared thumbnail cache for plant visualization scripts.

Generates thumbnails once and caches them to disk, avoiding expensive
re-generation across multiple scripts.
"""

import base64
import json
from pathlib import Path
from typing import Dict, Optional

import cv2
from tqdm import tqdm


class ThumbnailCache:
    """Persistent thumbnail cache with base64 encoding."""

    def __init__(self, cache_dir: str = 'thumbnails_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / 'thumbnails.json'
        self._cache: Dict[str, str] = {}
        self._load_cache()

    def _load_cache(self):
        """Load existing cache from disk."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                self._cache = json.load(f)
            print(f"Loaded {len(self._cache)} cached thumbnails")

    def _save_cache(self):
        """Save cache to disk."""
        with open(self.cache_file, 'w') as f:
            json.dump(self._cache, f)

    def _create_thumbnail(self, image_path: str, max_size: int) -> Optional[str]:
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

    def _get_cache_key(self, image_path: str, max_size: int) -> str:
        """Generate cache key from image path and size."""
        # Use absolute path to ensure consistency
        abs_path = str(Path(image_path).resolve())
        return f"{abs_path}:{max_size}"

    def get(self, image_path: str, max_size: int = 200) -> Optional[str]:
        """
        Get thumbnail from cache or generate if not cached.

        Args:
            image_path: Path to the image
            max_size: Maximum dimension in pixels

        Returns:
            Base64-encoded thumbnail or None
        """
        cache_key = self._get_cache_key(image_path, max_size)

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Generate and cache
        thumbnail = self._create_thumbnail(image_path, max_size)
        if thumbnail:
            self._cache[cache_key] = thumbnail
            # Save periodically (every 100 new thumbnails)
            if len(self._cache) % 100 == 0:
                self._save_cache()

        return thumbnail

    def get_batch(
        self,
        image_paths: list[str],
        max_size: int = 200,
        show_progress: bool = True
    ) -> Dict[str, str]:
        """
        Get thumbnails for multiple images.

        Args:
            image_paths: List of image paths
            max_size: Maximum dimension in pixels
            show_progress: Show progress bar

        Returns:
            Dictionary mapping image_path to base64 thumbnail
        """
        thumbnails = {}

        iterator = tqdm(image_paths, desc="Generating thumbnails") if show_progress else image_paths

        for path in iterator:
            thumb = self.get(path, max_size)
            if thumb:
                thumbnails[path] = thumb

        # Save cache after batch operation
        self._save_cache()

        return thumbnails

    def clear(self):
        """Clear all cached thumbnails."""
        self._cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        print("Thumbnail cache cleared")

    def __len__(self):
        """Return number of cached thumbnails."""
        return len(self._cache)
