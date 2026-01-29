import argparse
import os
import sys
from glob import glob
from pathlib import Path

from PIL import Image, ImageFilter
from tqdm import tqdm


def fix_relative_path(path):
    # pathlibで統一して簡潔にする
    return str(Path(path).expanduser().resolve())


def sorted_glob(patterns):
    files = []
    for pattern in patterns:
        files.extend(glob(pattern, recursive=True))
    return sorted(files)


def main(input_dir: str, out_dir: str, input_format: str, output_format: str):
    # Build case-insensitive glob patterns based on input format
    fmt = input_format.lower()
    resolved_input_dir = fix_relative_path(input_dir)
    if fmt == 'all':
        img_patterns = [
            os.path.join(resolved_input_dir, '**', '*.[jJ][pP][gG]'),
            os.path.join(resolved_input_dir, '**', '*.[jJ][pP][eE][gG]'),
            os.path.join(resolved_input_dir, '**', '*.[pP][nN][gG]'),
        ]
    elif fmt in ('jpg', 'jpeg'):
        img_patterns = [
            os.path.join(resolved_input_dir, '**', '*.[jJ][pP][gG]'),
            os.path.join(resolved_input_dir, '**', '*.[jJ][pP][eE][gG]'),
        ]
    elif fmt == 'png':
        img_patterns = [
            os.path.join(resolved_input_dir, '**', '*.[pP][nN][gG]'),
        ]
    else:
        # Fallback: use specified format as-is (case-sensitive)
        img_patterns = [
            os.path.join(resolved_input_dir, '**', f'*.{input_format}'),
        ]
    imgs = sorted_glob(img_patterns)
    os.makedirs(out_dir, exist_ok=True)

    Image.MAX_IMAGE_PIXELS = 933120000
    isatty = sys.stdout.isatty()
    pbar = tqdm(
        total=len(imgs),
        unit="file",
        leave=False,
        dynamic_ncols=isatty,
        disable=not isatty,
    )
    for img_path in imgs:
        pbar.set_description(os.path.basename(img_path))
        if not isatty:
            print(f"Processing {os.path.basename(img_path)}")

        image = Image.open(img_path)

        sharpened_image = image.filter(
            ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
        )
        relative_path = os.path.relpath(img_path, start=Path(input_dir))
        save_path = os.path.join(
            out_dir, os.path.splitext(relative_path)[0] + f".{output_format}"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        sharpened_image.save(save_path)
        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input image or folder"
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Output folder")
    parser.add_argument(
        "--input-format", type=str, default="all", help="Input image extension: all | jpg | jpeg | png"
    )
    parser.add_argument(
        "--output-format", type=str, default="png", help="Output image extension"
    )

    args = parser.parse_args()
    args.input = fix_relative_path(args.input)
    args.output = fix_relative_path(args.output)

    main(
        input_dir=args.input,
        out_dir=args.output,
        input_format=args.input_format,
        output_format=args.output_format,
    )
