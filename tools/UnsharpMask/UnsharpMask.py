import os
import argparse
from glob import glob
from pathlib import Path

from tqdm import tqdm
from PIL import Image, ImageFilter


def fix_relative_path(path):
  if os.path.isabs(path):
    return path
  else:
    return os.path.join('.', os.path.relpath(path, start=Path('.')))


def sorted_glob(patterns):
  files = []
  for pattern in patterns:
    files.extend(glob(pattern, recursive=True))
  return sorted(files)


def main(input_dir='./input/', out_dir='./output/'):
  img_patterns = [
      os.path.join(os.path.expanduser(input_dir), '**', '*.jpg'),
      os.path.join(os.path.expanduser(input_dir), '**', '*.png')
  ]
  imgs = sorted_glob(img_patterns)
  os.makedirs(out_dir, exist_ok=True)

  pbar = tqdm(total=len(imgs), desc="UnsharpMask Processing", unit="item")
  for img_path in imgs:
    image = Image.open(img_path)

    sharpened_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    relative_path = os.path.relpath(img_path, start=Path(input_dir))
    save_path = os.path.join(out_dir, relative_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sharpened_image.save(save_path)
    pbar.update(1)

  pbar.close()


if __name__ == '__main__':
  # Parse command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i', '--input', type=str, default='input', help='Input image or folder'
  )
  parser.add_argument(
      '-o', '--output', type=str, default='output', help='Output folder'
  )

  args = parser.parse_args()
  args.input = os.path.expanduser(fix_relative_path(args.input))
  args.output = os.path.expanduser(fix_relative_path(args.output))

  main(input_dir=args.input, out_dir=args.output)
