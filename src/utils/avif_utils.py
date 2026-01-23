"""
avif_utils.py

python -m src.utils.avif_utils --input /home/toantim/ToanFolder/Hand-gesture-controlled/data/hagrid-sample-30k-384p/hagrid_30k/train_val_call  --quality 80

This script provides utilities for converting images to AVIF format.

Features:
- Convert a single image (JPG/PNG) to AVIF
- Recursively convert all images in a folder (and subfolders) to AVIF
- Recreate the original folder structure in a new "_compressed" folder
- Optional lossless compression
- Optional resizing before conversion
- Optional folder size comparison (before and after conversion)
- Command-line interface (CLI) for easy usage

Usage examples:
- Convert a single image:
    python avif_utils.py --input input.jpg --output output.avif --quality 85

- Convert all images in a folder:
    python avif_utils.py --input /path/to/folder --quality 80

- Use lossless compression:
    python avif_utils.py --input input.png --lossless

- Resize before conversion:
    python avif_utils.py --input input.jpg --resize 800x600

- Show folder size before and after conversion:
    python avif_utils.py --input /path/to/folder --quality 80 --show-size
"""

"""
Folder naming rules:
- Create a new folder in the same directory as the original parent folder:
  original_folder_name + "_compressed"
- Inside that folder, recreate the same folder structure as the original.
- If a folder contains images, compress them and save them in the corresponding new folder.
- If a folder contains no images, just recreate the folder structure without changes.
- If a subfolder contains images, the new folder name should be:
  original_subfolder_name + "_compressed"

Image formats: jpg, jpeg, png, gif, bmp, webp
Compression: Uses Pillow for JPEG/PNG compression.

The script prints a summary of:
- Total images found
- Total images compressed
- Total size saved

Original files are never modified.
"""



from PIL import Image
import os
from pathlib import Path
import argparse

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def convert_to_avif(input_path: str, output_path: str, quality: int = 80, lossless: bool = False, resize: tuple = None):
    """
    Convert an image to AVIF format.
    """
    img = Image.open(input_path).convert("RGB")

    if resize:
        img = img.resize(resize, Image.ANTIALIAS)

    img.save(
        output_path,
        format="AVIF",
        quality=quality,
        lossless=lossless
    )


def folder_size_in_bytes(folder_path: str):
    """
    Calculate total folder size in bytes.
    """
    total = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            total += os.path.getsize(os.path.join(root, file))
    return total


def human_readable_size(size, decimal_places=2):
    """
    Convert bytes to human-readable format.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.{decimal_places}f} {unit}"
        size /= 1024
    return f"{size:.{decimal_places}f} PB"


def get_compressed_folder_name(folder_name: str) -> str:
    """
    Return compressed folder name.
    """
    return f"{folder_name}_compressed"


def compress_recursive(base_input: Path, base_output: Path, quality: int, lossless: bool, resize: tuple):
    """
    Recursively compress images and save them into new folder structure.
    """
    total_images = 0
    total_compressed = 0
    total_saved_bytes = 0

    # First count total images for progress tracking
    total_to_process = 0
    for root, _, files in os.walk(base_input):
        total_to_process += sum(1 for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS)

    processed = 0

    for root, dirs, files in os.walk(base_input):
        root_path = Path(root)
        relative_root = root_path.relative_to(base_input)

        # Create corresponding output folder
        output_root = base_output / relative_root
        output_root.mkdir(parents=True, exist_ok=True)

        # Process image files
        image_files = [f for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS]

        if image_files:
            print(f"Processing folder: {root_path}")

            for file_name in image_files:
                processed += 1

                if processed % 100 == 0 or processed == total_to_process:
                    print(f"  [{processed}/{total_to_process}] Compressing: {file_name}")

                total_images += 1

                input_file_path = root_path / file_name
                output_file_path = output_root / f"{Path(file_name).stem}.avif"

                original_size = input_file_path.stat().st_size
                convert_to_avif(str(input_file_path), str(output_file_path),
                                quality=quality, lossless=lossless, resize=resize)

                compressed_size = output_file_path.stat().st_size
                saved = max(0, original_size - compressed_size)

                if saved > 0:
                    total_compressed += 1
                    total_saved_bytes += saved

    return total_images, total_compressed, total_saved_bytes


def main():
    parser = argparse.ArgumentParser(description="Recursive AVIF image compressor")
    parser.add_argument("--input", type=str, required=True, help="Input folder path")
    parser.add_argument("--quality", type=int, default=80, help="AVIF quality (0-100)")
    parser.add_argument("--lossless", action="store_true", help="Use lossless compression")
    parser.add_argument("--resize", type=str, default=None, help="Resize to WIDTHxHEIGHT (e.g., 800x600)")

    args = parser.parse_args()

    input_folder = Path(args.input).resolve()
    if not input_folder.is_dir():
        print("Invalid folder path.")
        return

    # Create base output folder
    base_output = input_folder.parent / get_compressed_folder_name(input_folder.name)

    resize = None
    if args.resize:
        w, h = args.resize.lower().split("x")
        resize = (int(w), int(h))

    # Run compression
    total_images, total_compressed, total_saved_bytes = compress_recursive(
        input_folder, base_output, args.quality, args.lossless, resize
    )

    # Summary
    print("\nSummary:")
    print(f"Total images found: {total_images}")
    print(f"Total images compressed: {total_compressed}")
    print(f"Total size saved: {human_readable_size(total_saved_bytes)}")


if __name__ == "__main__":
    main()
