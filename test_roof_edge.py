import os
import argparse

import numpy as np
from PIL import Image
import cv2

from src.createmodel.housemodeling.roof_edge_detection import RoofEdgeDetection


def draw_edges_on_image(dsm_grid_rgbs, corners, edges):
  """
  画像に頂点を線でつなぐ。

  Args:
    dsm_grid_rgbs (np.array): RGB画像データ
    corners (np.array): 頂点の位置のリスト (num of corners, 2)
    edges (np.array): 頂点の番号の組のリスト (num of edges, 2)

  Returns:
    np.array: 線が描画された画像
  """

  # RGB画像データをBGRに変換（OpenCVはBGRを使用）
  image = cv2.cvtColor(dsm_grid_rgbs, cv2.COLOR_RGB2BGR)

  # edges に従って線を描画
  for edge in edges:
    start_point = tuple(corners[edge[0]])
    end_point = tuple(corners[edge[1]])
    cv2.line(image, start_point, end_point, (0, 255, 0), 2)  # 緑色の線、太さ2

  return image


def main():
  parser = argparse.ArgumentParser(description="Search LAS files by bounding box.")
  parser.add_argument("input_image_path", type=str, help="Input Image for drawing roof edge")
  parser.add_argument("--output_dir", "-o", type=str, help="Optional directory for output image")

  args = parser.parse_args()

  # python3 test_roof_edge.py ~/bldg-lod2-tool/tools/Real-ESRGAN/output/DSM_-214_-347_out.jpg -o .

  input_image_path: str = os.path.expanduser(args.input_image_path)

  roof_edge_detection = RoofEdgeDetection(
      os.path.expanduser('~/bldg-lod2-tool/src/createmodel/data/roof_edge_detection_parameter.pth'),
      True,
  )

  print(input_image_path)
  image = Image.open(input_image_path)

  if image.mode != 'RGB':
    image = image.convert('RGB')

  _image_size = 256
  square_dsm_grid_rgbs = Image.new('RGB', (_image_size, _image_size), "white")
  width, height = image.size
  top = (_image_size - height) // 2
  left = (_image_size - width) // 2
  if width <= _image_size and height <= _image_size:
    square_dsm_grid_rgbs.paste(image, (left, top))
  else:
    image = image.resize((_image_size, _image_size), Image.ANTIALIAS)
    square_dsm_grid_rgbs = image

  dsm_grid_rgbs = np.array(square_dsm_grid_rgbs)
  corners, edges = roof_edge_detection.infer(dsm_grid_rgbs)

  image_with_edges = draw_edges_on_image(dsm_grid_rgbs, corners, edges)

  input_image_basename = os.path.basename(input_image_path)
  input_image_basename_without_ext, _ = os.path.splitext(input_image_basename)
  output_image_basename = f"{input_image_basename_without_ext}_out.png"
  if args.output_dir is not None:
    output_dir = os.path.expanduser(args.output_dir)
    output_image_path = os.path.join(output_dir, output_image_basename)
  else:
    input_image_dir = os.path.dirname(input_image_path)
    output_image_path = os.path.join(input_image_dir, output_image_basename)

  cv2.imwrite(output_image_path, image_with_edges)
  print(output_image_path)


if __name__ == '__main__':
  main()
