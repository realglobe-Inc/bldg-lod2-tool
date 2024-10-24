import os
import argparse
from typing import Union

import numpy as np
from PIL import Image
import laspy


def make_image(las_path: str, bbox: tuple[float], output_dir: Union[str, None]):
  """
  Make Image with DSM (LAS) File
  """
  las = laspy.read(las_path)

  grid_distance = 0.25

  x_min, y_min, x_max, y_max = bbox
  mask = (las.x >= x_min) & (las.x <= x_max) & (las.y >= y_min) & (las.y <= y_max)
  filtered_points = las.points[mask]
  points = np.stack(
      [
          filtered_points.x,
          filtered_points.y,
          filtered_points.z,
          filtered_points.red,
          filtered_points.green,
          filtered_points.blue,
      ],
      axis=0,
  ).transpose((1, 0))

  width = int(np.floor(x_max / grid_distance - x_min / grid_distance + 0.0001))
  height = int(np.floor(y_max / grid_distance - y_min / grid_distance + 0.0001))

  image_data = np.zeros((width, height, 3), dtype=np.uint8)

  for point in points:
    x, y, _z, r, g, b = point
    x_pos = int(np.floor((x - x_min) / grid_distance + 0.0001))
    y_pos = int(np.floor((y - y_min) / grid_distance + 0.0001))
    image_data[x_pos, y_pos] = [r / 256, g / 256, b / 256]

  print(las_path)
  las_basename = os.path.basename(las_path)
  las_basename_without_ext, _ = os.path.splitext(las_basename)
  image_basename = f"{las_basename_without_ext}_{x_min},{y_min},{x_max},{y_max}.png"
  image = Image.fromarray(image_data, "RGB")
  if output_dir is not None:
    output_dir = os.path.expanduser(output_dir)
    image_path = os.path.join(output_dir, image_basename)
  else:
    las_dir = os.path.dirname(las_path)
    image_path = os.path.join(las_dir, image_basename)

  image.save(image_path)
  print(image_path)


def is_bbox_overlap(bbox1: tuple[float], bbox2: tuple[float]):
  """
  Check if two bounding boxes overlap.
  bbox1 and bbox2: (min_x, min_y, max_x, max_y)
  """
  return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])


def get_las_bbox(las_file: str) -> Union[tuple[float], None]:
  """
  Get the bounding box (min_x, min_y, max_x, max_y) of the LAS file.
  """
  try:
    with laspy.open(las_file) as f:
      header = f.header
      min_x, max_x = header.mins[0], header.maxs[0]
      min_y, max_y = header.mins[1], header.maxs[1]
      return (min_x, min_y, max_x, max_y)
  except Exception as e:
    print(f"Error reading {las_file}: {e}")
    return None


def search_dsm_files(bbox: tuple[float], dsm_dir: str):
  """
  Search for LAS files that overlap with the given bounding box.
  """
  matching_files: list[str] = []

  for root, _dirs, files in os.walk(dsm_dir):
    for file in files:
      if file.endswith(".las"):
        file_path = os.path.join(root, file)
        las_bbox = get_las_bbox(file_path)
        if las_bbox is None:
          continue

        if las_bbox and is_bbox_overlap(las_bbox, bbox):
          matching_files.append(file_path)

  return matching_files


def main():
  """X,Y 座標領域に入っている DSM の RGB を png に保存"""
  parser = argparse.ArgumentParser(description="Search LAS files by bounding box.")
  parser.add_argument("min_x", type=float, help="Minimum X coordinate")
  parser.add_argument("min_y", type=float, help="Minimum Y coordinate")
  parser.add_argument("max_x", type=float, help="Maximum X coordinate")
  parser.add_argument("max_y", type=float, help="Maximum Y coordinate")
  parser.add_argument("dsm_dir", type=str, help="Directory containing DSM (LAS) files")
  parser.add_argument("--output_dir", "-o", type=str, help="Optional directory for output image")

  args = parser.parse_args()

  # python3 search_dsm_file_by_pos.py -21146 -34454 -21132 -34440 ~/DSM/DSM/

  bbox: tuple[float] = (args.min_x, args.min_y, args.max_x, args.max_y)
  matching_files = search_dsm_files(bbox, args.dsm_dir)

  if matching_files:
    for file in matching_files:
      make_image(file, bbox, args.output_dir)
  else:
    print("No matching LAS files found.")


if __name__ == "__main__":
  main()
