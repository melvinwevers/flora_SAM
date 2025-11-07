#!/usr/bin/env python3
"""
Batch Interactive SAM segmentation tool
Process multiple illustration images from flora_meta.xlsx
- Shows images one by one for interactive segmentation
- Saves masks with proper naming
- Skips images with existing masks
- Allows navigation between images
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm


def download_sam_model(model_type="vit_b", save_path=None):
    """
    Download SAM model checkpoint if it doesn't exist.

    Args:
        model_type: One of 'vit_b', 'vit_l', 'vit_h'
        save_path: Path to save the model. If None, save in current directory.

    Returns:
        Path to the downloaded model
    """
    # SAM model URLs
    MODEL_URLS = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }

    if model_type not in MODEL_URLS:
        raise ValueError(
            f"Invalid model_type: {model_type}. Choose from {list(MODEL_URLS.keys())}"
        )

    if save_path is None:
        save_path = f"sam_{model_type}.pth"

    save_path = Path(save_path)

    # Check if model already exists
    if save_path.exists():
        print(f"Model already exists at {save_path}")
        return save_path

    # Download the model
    url = MODEL_URLS[model_type]
    print(f"Downloading SAM model ({model_type}) from {url}...")
    print(f"Saving to: {save_path}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(save_path, "wb") as f,
            tqdm(
                desc=f"Downloading {model_type}",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"Model downloaded successfully to {save_path}")
        return save_path

    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise Exception(f"Failed to download model: {e}")


class BatchInteractiveSAM:
    def __init__(
        self, data_dir, meta_file, model_path, mask_output_dir, model_type="vit_b"
    ):
        # Load metadata
        print(f"Loading metadata from {meta_file}...")
        df = pd.read_excel(meta_file)
        self.illustrations = df[df["File type"] == "illustration"][
            "File name"
        ].values.tolist()
        print(f"Found {len(self.illustrations)} illustration images")

        self.data_dir = Path(data_dir)
        self.mask_output_dir = Path(mask_output_dir)
        self.mask_output_dir.mkdir(exist_ok=True)

        # Metadata file for tracking masks
        self.metadata_file = self.mask_output_dir / "masks_metadata.json"
        self.masks_metadata = self.load_metadata()

        # Initialize SAM model
        print(f"Loading SAM model ({model_type})...")

        # Download model if it doesn't exist
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"Model not found at {model_path}")
            model_path = download_sam_model(model_type, save_path=model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        sam = sam_model_registry[model_type](checkpoint=str(model_path))
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        print("SAM model loaded successfully!")

        # State
        self.current_index = 0
        self.current_image = None
        self.original_image = None
        self.cropped_image = None  # Cropped version for display
        self.display_original = None
        self.display_image = None
        self.scale_factor = 1.0
        self.crop_percent = 0.05  # Crop 5% from each edge for display
        self.crop_offset_x = 0
        self.crop_offset_y = 0
        self.points = []
        self.labels = []
        self.current_mask = None
        self.window_name = (
            "Batch SAM - Click BACKGROUND | z: undo | s: save | "
            "b: back | n: next | r: reset | q: quit"
        )

        # Window setup
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def load_metadata(self):
        """Load existing metadata file or create new dict"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                print(f"Loaded existing metadata with {len(data)} entries")
                return data
            except Exception as e:
                print(f"Warning: Could not load metadata file: {e}")
                return {}
        return {}

    def save_metadata(self):
        """Save metadata to JSON file"""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.masks_metadata, f, indent=2)
            print(f"✓ Metadata saved to {self.metadata_file}")
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def get_mask_path(self, image_name):
        """Get the path for the mask file"""
        base_name = Path(image_name).stem  # Remove .jpg extension
        mask_name = f"{base_name}_mask.png"
        return self.mask_output_dir / mask_name

    def has_existing_mask(self, image_name):
        """Check if mask already exists for this image"""
        return self.get_mask_path(image_name).exists()

    def find_image_path(self, image_name):
        """Find image across all volumes (1-28)"""
        # Try to find the image in any volume directory
        for volume in range(1, 29):  # Volumes 1 through 28
            image_path = self.data_dir / str(volume) / image_name
            if image_path.exists():
                return image_path
        return None

    def load_image(self, image_name):
        """Load and prepare image for display"""
        image_path = self.find_image_path(image_name)

        if image_path is None:
            print(f"Warning: Image not found in any volume: {image_name}")
            return False

        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            print(f"Warning: Could not load image {image_path}")
            return False

        image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        # Set FULL original image in predictor (uncropped)
        self.predictor.set_image(image_rgb)

        # Create cropped version for display (crop edges to avoid background clicks)
        h, w = self.original_image.shape[:2]
        crop_h = int(h * self.crop_percent)
        crop_w = int(w * self.crop_percent)

        self.crop_offset_y = crop_h
        self.crop_offset_x = crop_w

        self.cropped_image = self.original_image[
            crop_h:h - crop_h, crop_w:w - crop_w
        ].copy()

        # Scale cropped image to fit screen if needed
        self.scale_factor = 1.0
        screen_height = 900
        screen_width = 1600

        ch, cw = self.cropped_image.shape[:2]
        if ch > screen_height or cw > screen_width:
            scale_h = screen_height / ch
            scale_w = screen_width / cw
            self.scale_factor = min(scale_h, scale_w)

            new_w = int(cw * self.scale_factor)
            new_h = int(ch * self.scale_factor)

            self.display_original = cv2.resize(
                self.cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
        else:
            self.display_original = self.cropped_image.copy()

        self.display_image = self.display_original.copy()

        # Reset state
        self.points = []
        self.labels = []
        self.current_mask = None

        # Automatically add a positive point in the upper right corner
        # This gives a good starting segmentation of the background
        auto_x = int(w * 0.95)  # 95% to the right
        auto_y = int(h * 0.05)  # 5% from top
        self.add_point(auto_x, auto_y, positive=True)
        print(f"Auto-added background point at upper right corner ({auto_x}, {auto_y})")

        return True

    def show_current_image(self):
        """Display current image with info"""
        if self.current_index >= len(self.illustrations):
            print("All images processed!")
            return False

        image_name = self.illustrations[self.current_index]

        # Check if mask exists
        has_mask = self.has_existing_mask(image_name)
        status = "[DONE]" if has_mask else "[TODO]"

        print(f"\n{'='*60}")
        print(f"Image {self.current_index + 1}/{len(self.illustrations)} {status}")
        print(f"File: {image_name}")
        if has_mask:
            print(f"Existing mask: {self.get_mask_path(image_name)}")
        print(f"{'='*60}")

        if not self.load_image(image_name):
            return False

        if self.display_original is not None and self.display_image is not None:
            cv2.resizeWindow(
                self.window_name,
                self.display_original.shape[1],
                self.display_original.shape[0],
            )
            cv2.imshow(self.window_name, self.display_image)

        return True

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Scale coordinates back to cropped image size
            cropped_x = int(x / self.scale_factor)
            cropped_y = int(y / self.scale_factor)

            # Convert from cropped to original image coordinates
            orig_x = cropped_x + self.crop_offset_x
            orig_y = cropped_y + self.crop_offset_y

            # If Shift is pressed, make it a negative point
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.add_point(orig_x, orig_y, positive=False)
            else:
                self.add_point(orig_x, orig_y, positive=True)

    def add_point(self, x, y, positive=True):
        """Add a point and update segmentation"""
        self.points.append([x, y])
        self.labels.append(1 if positive else 0)

        label = "positive" if positive else "negative"
        print(f"Added {label} point at ({x}, {y})")

        self.predict_and_display()

    def undo_last_point(self):
        """Remove the last added point"""
        if len(self.points) > 0:
            removed_point = self.points.pop()
            removed_label = self.labels.pop()
            label_type = "positive" if removed_label == 1 else "negative"
            print(f"Undid {label_type} point at {removed_point}")

            # Re-predict if there are still points
            if len(self.points) > 0:
                self.predict_and_display()
            else:
                # No points left, clear mask and show original image
                self.current_mask = None
                if self.display_original is not None:
                    self.display_image = self.display_original.copy()
                    cv2.imshow(self.window_name, self.display_image)
        else:
            print("No points to undo")

    def predict_and_display(self):
        """Generate mask and update display"""
        if len(self.points) == 0:
            return

        # Predict
        input_points = np.array(self.points)
        input_labels = np.array(self.labels)

        masks, scores, logits = self.predictor.predict(
            point_coords=input_points, point_labels=input_labels, multimask_output=True
        )

        # Use the medium mask (index 1) - balanced between aggressive and precise
        # SAM returns masks in order: [large/whole, medium, small/part]
        # Change to masks[0] for more aggressive, masks[2] for more precise
        best_mask = masks[1]

        # Store the background mask (for visualization)
        # We'll invert it when saving to get the object
        self.current_mask = best_mask

        print(
            f"Background mask generated! Score: {scores[1]:.3f}, "
            f"Pixels: {np.sum(best_mask)}"
        )

        # Update display
        self.update_display()

    def update_display(self):
        """Update the display with points and mask overlay"""
        if self.original_image is None or self.display_original is None:
            return

        full_image = self.original_image.copy()

        # Draw mask overlay on full resolution image
        if self.current_mask is not None:
            overlay = full_image.copy()
            overlay[self.current_mask] = (
                overlay[self.current_mask] * 0.5 + np.array([0, 0, 255]) * 0.5
            )
            full_image = overlay.astype(np.uint8)

        # Draw points on full resolution image
        for point, label in zip(self.points, self.labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            radius = max(3, int(8 / self.scale_factor))
            thickness = max(1, int(2 / self.scale_factor))
            cv2.circle(full_image, tuple(point), radius, color, -1)
            cv2.circle(full_image, tuple(point), radius, (255, 255, 255), thickness)

        # Crop the image for display
        h, w = self.original_image.shape[:2]
        crop_h = int(h * self.crop_percent)
        crop_w = int(w * self.crop_percent)
        cropped_display = full_image[crop_h:h - crop_h, crop_w:w - crop_w]

        # Scale for display if needed
        if self.scale_factor != 1.0:
            disp_h, disp_w = self.display_original.shape[:2]
            self.display_image = cv2.resize(
                cropped_display, (disp_w, disp_h), interpolation=cv2.INTER_AREA
            )
        else:
            self.display_image = cropped_display

        cv2.imshow(self.window_name, self.display_image)

    def save_current_mask(self):
        """Save the current mask and update metadata"""
        if self.current_mask is None:
            print("No mask to save! Please add points first.")
            return False

        if self.original_image is None:
            print("Error: No image loaded!")
            return False

        image_name = self.illustrations[self.current_index]
        mask_path = self.get_mask_path(image_name)

        # Invert the mask - current_mask is background, we want to save the object
        object_mask = ~self.current_mask

        # Exclude the cropped border areas from the final mask
        # Set border regions to False (excluded)
        h, w = self.original_image.shape[:2]
        crop_h = int(h * self.crop_percent)
        crop_w = int(w * self.crop_percent)

        # Create final mask with borders excluded (ensure numpy array)
        final_object_mask = np.array(object_mask, copy=True)
        final_object_mask[:crop_h, :] = False  # Top border
        final_object_mask[h - crop_h:, :] = False  # Bottom border
        final_object_mask[:, :crop_w] = False  # Left border
        final_object_mask[:, w - crop_w:] = False  # Right border

        # Calculate mask area (number of pixels) for the object (excluding borders)
        mask_area = int(np.sum(final_object_mask))

        # Save the segmented object (original size, with borders excluded)
        segmented = self.original_image.copy()
        segmented[~final_object_mask] = 0
        segmented_path = str(mask_path).replace("_mask.png", "_segmented.png")
        cv2.imwrite(segmented_path, segmented)
        print(f"✓ Segmented object saved to {segmented_path} (borders excluded)")

        # Update metadata
        mask_entry = {
            "image": image_name,
            "mask_file": mask_path.name,
            "segmented_file": Path(segmented_path).name,
            "area_pixels": mask_area,
            "image_dimensions": {
                "height": h,
                "width": w,
            },
        }

        # Use mask filename as key for easy lookup
        self.masks_metadata[mask_path.name] = mask_entry
        self.save_metadata()

        print(f"✓ Metadata updated (area: {mask_area} pixels)")

        return True

    def next_image(self):
        """Move to next image"""
        self.current_index += 1
        return self.show_current_image()

    def previous_image(self):
        """Move to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            return self.show_current_image()
        else:
            print("Already at first image")
            return True

    def skip_to_next_unprocessed(self):
        """Skip to the next image without an existing mask"""
        while self.current_index < len(self.illustrations):
            image_name = self.illustrations[self.current_index]
            if not self.has_existing_mask(image_name):
                return self.show_current_image()
            self.current_index += 1

        # If we've reached the end, go back to start
        if self.current_index >= len(self.illustrations):
            print("\nAll images have been processed!")
            return False

        return self.show_current_image()

    def run(self):
        """Main loop"""
        print("\n" + "=" * 60)
        print("BATCH INTERACTIVE SAM SEGMENTATION")
        print("=" * 60)
        print("\nControls:")
        print("  Click: Add positive point (include area)")
        print("  Shift+Click: Add negative point (exclude area)")
        print("  'z': Undo last point")
        print("  's': Save mask and move to next image")
        print("  'n': Skip to next image without saving")
        print("  'r': Reset all points for current image")
        print("  'u': Skip to next unprocessed image")
        print("  'q': Quit")
        print("=" * 60 + "\n")

        # Skip to first unprocessed image
        if not self.skip_to_next_unprocessed():
            print("No unprocessed images found. Starting from beginning...")
            self.current_index = 0
            if not self.show_current_image():
                return

        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nQuitting...")
                break

            elif key == ord("s"):
                # Save and move to next
                if self.save_current_mask():
                    if not self.next_image():
                        print("\nAll images processed! Press 'q' to quit.")

            elif key == ord("n"):
                # Skip to next without saving
                if not self.next_image():
                    print("\nReached end of images. Press 'q' to quit.")

            elif key == ord("p"):
                # Go back to previous image
                self.previous_image()

            elif key == ord("z"):
                # Undo last point
                self.undo_last_point()

            elif key == ord("r"):
                # Reset current image
                self.points = []
                self.labels = []
                self.current_mask = None
                # Re-add automatic point
                if self.original_image is not None:
                    h, w = self.original_image.shape[:2]
                    auto_x = int(w * 0.95)
                    auto_y = int(h * 0.05)
                    self.add_point(auto_x, auto_y, positive=True)
                    print(
                        f"Reset points and re-added auto point at ({auto_x}, {auto_y})"
                    )

            elif key == ord("u"):
                # Skip to next unprocessed
                if not self.skip_to_next_unprocessed():
                    print("\nNo more unprocessed images. Press 'q' to quit.")

        cv2.destroyAllWindows()

        # Print summary
        processed_count = sum(
            1 for img in self.illustrations if self.has_existing_mask(img)
        )
        print(f"\n{'='*60}")
        print(f"SUMMARY: {processed_count}/{len(self.illustrations)} images processed")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch interactive SAM segmentation tool"
    )
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument(
        "--meta", default="flora_meta.xlsx", help="Path to metadata Excel file"
    )
    parser.add_argument(
        "--model", default="sam_vit_b.pth", help="Path to SAM model checkpoint"
    )
    parser.add_argument(
        "--model-type",
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model type",
    )
    parser.add_argument("--output", default="masks", help="Output directory for masks")

    args = parser.parse_args()

    try:
        app = BatchInteractiveSAM(
            data_dir=args.data_dir,
            meta_file=args.meta,
            model_path=args.model,
            mask_output_dir=args.output,
            model_type=args.model_type,
        )
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
