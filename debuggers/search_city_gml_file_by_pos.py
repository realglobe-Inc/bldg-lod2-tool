import json
import os
import argparse
import xml.etree.ElementTree as ET

from tqdm import tqdm
from pyproj import Transformer


def is_bbox_overlap(bbox1: tuple[float], bbox2: tuple[float]):
  """
  Check if two bounding boxes overlap.
  bbox1 and bbox2: (min_x, min_y, max_x, max_y)
  """
  return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])


def to_cartesian_point(lat: float, lon: float, dst_epsg: int):
  """
  Convert latitude and longitude to the target Cartesian coordinate system.
  """
  try:
    transformer = Transformer.from_crs("epsg:6668", f"epsg:{dst_epsg}", always_xy=True)
    # 6668 (緯度経度座標) から、6677などの平面直角座標へ変換
    x, y = transformer.transform(float(lon), float(lat))
    return x, y
  except Exception as e:
    print(f"Error converting coordinates: {e}")
    return None, None


def get_gml_bbox(gml_file: str, dst_epsg: int, cache_file: str, cache: dict[str, list[str]] = {}):
  """
  GMLファイルのバウンディングボックスを取得し、キャッシュファイルに保存する。
  キャッシュが既に存在すればそれを使用する。
  """
  # GMLファイルに対してキャッシュが存在するか確認
  if gml_file in cache:
    return tuple(cache[gml_file])

  try:
    tree = ET.parse(gml_file)
    root = tree.getroot()

    ns = {"gml": "http://www.opengis.net/gml"}
    lower_corner = root.find(".//gml:lowerCorner", ns)
    upper_corner = root.find(".//gml:upperCorner", ns)

    if lower_corner is not None and upper_corner is not None:
      lower_lat, lower_lon = lower_corner.text.split(" ")[:2]
      upper_lat, upper_lon = upper_corner.text.split(" ")[:2]

      lower_x, lower_y = to_cartesian_point(float(lower_lat), float(lower_lon), dst_epsg)
      upper_x, upper_y = to_cartesian_point(float(upper_lat), float(upper_lon), dst_epsg)
      bbox = (lower_x, lower_y, upper_x, upper_y)

      # キャッシュに新しいデータを追加
      cache[gml_file] = bbox
      with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)

      return bbox
    else:
      return None

  except Exception as e:
    print(f"Error reading {gml_file}: {e}")
    return None


def search_city_gml_files(bbox: tuple[float], directory: str, dst_epsg: int):
  """
  Search for GML files that overlap with the given bounding box.
  """
  matching_files: list[str] = []

  for root, dirs, files in os.walk(directory):
    pbar = tqdm(total=len(files), unit='item', desc=f'Processing {root}')

    cache_file = os.path.join(root, "xy_min_max.json")
    cache = {}
    if os.path.exists(cache_file):
      with open(cache_file, "r") as f:
        cache = json.load(f)

    for file in files:
      if file.endswith(".gml"):
        file_path = os.path.join(root, file)
        gml_bbox: tuple[int] = get_gml_bbox(file_path, dst_epsg, cache_file, cache)
        if gml_bbox and is_bbox_overlap(gml_bbox, bbox):
          matching_files.append(file_path)

      pbar.update(1)

  return matching_files


def main():
  parser = argparse.ArgumentParser(description="Search GML files by bounding box.")
  parser.add_argument("min_x", type=float, help="Minimum X coordinate")
  parser.add_argument("min_y", type=float, help="Minimum Y coordinate")
  parser.add_argument("max_x", type=float, help="Maximum X coordinate")
  parser.add_argument("max_y", type=float, help="Maximum Y coordinate")
  parser.add_argument("city_gml_dir", type=str, help="Directory containing CityGML files")
  parser.add_argument("dst_epsg", type=int, help="EPSG code for cartesian coordinate")

  args = parser.parse_args()
  # python3 search_city_gml_file_by_pos.py -9000 -51000 -10000 -52000 \
  #   ~/lod2_data/kawazaki/08_CityGML 6677

  bbox: tuple[float] = (args.min_x, args.min_y, args.max_x, args.max_y)
  city_gml_dir: str = os.path.expanduser(args.city_gml_dir)
  matching_files = search_city_gml_files(bbox, city_gml_dir, args.dst_epsg)

  if matching_files:
    print("Matching GML files:")
    for file in matching_files:
      print(file)
  else:
    print("No matching GML files found.")


if __name__ == "__main__":
  main()
