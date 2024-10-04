

import copy
import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from collections import deque


# 画像を読み込む
image_path = "/home/ubuntu/bldg-lod2-tool/debug/bldg-308a2cd7-131a-4e56-8fb6-9a91147ff7a1/layer.png"

# 色範囲のリスト（BGRフォーマット）
color_ranges = [
    (np.array([0, 0, 0]), np.array([0, 0, 0])),
    (np.array([0, 128, 64]), np.array([0, 128, 64])),
    (np.array([0, 192, 192]), np.array([0, 192, 192])),
    (np.array([0, 255, 0]), np.array([0, 255, 0])),
    (np.array([128, 64, 192]), np.array([128, 64, 192])),
    (np.array([128, 128, 64]), np.array([128, 128, 64])),
    (np.array([192, 88, 48]), np.array([192, 88, 48])),
    (np.array([255, 255, 0]), np.array([255, 255, 0])),
    (np.array([255, 255, 255]), np.array([255, 255, 255])),
]

import cv2
import numpy as np
from collections import deque

import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from collections import deque


# BFSによる領域探索
def bfs(image, start, visited):
  """BFSを使って同じ色でつながっている領域を探索"""
  h, w, _ = image.shape
  x_start, y_start = start
  color = tuple(image[x_start, y_start])  # 開始ピクセルの色
  queue = deque([start])
  region = []  # この領域のピクセル群
  visited[x_start, y_start] = True

  # 4方向の移動
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右

  while queue:
    x, y = queue.popleft()
    region.append((x, y))

    for dx, dy in directions:
      nx, ny = x + dx, y + dy

      # 範囲チェックと、まだ訪れていないか確認
      if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
        # 同じ色のピクセルか確認
        if tuple(image[nx, ny]) == color:
          visited[nx, ny] = True
          queue.append((nx, ny))

  return region


def segment_image_by_color(image):
  """画像を同じ色の領域に分割する"""
  h, w, _ = image.shape
  visited = np.zeros((h, w), dtype=bool)  # 訪問済みピクセルを記録
  regions = []  # 領域リスト

  for i in range(h):
    for j in range(w):
      if not visited[i, j]:
        # 新しい領域を発見したらBFSでその領域全体を探索
        region = bfs(image, (i, j), visited)
        regions.append(region)

  return regions


def simplify_polygons_and_save_image(image, step, prev_polygons=[]):
  """ポリゴンを単純化し、差分を計算して赤色で塗りつぶす"""
  current_image = image.copy()
  current_polygons = []

  # RGBで紫色の範囲を指定
  lower_color = np.array([255, 0, 0])  # 紫色の下限

  # BGRからRGBに変換
  rgb_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

  # 紫色の部分をマスク
  mask = cv2.inRange(rgb_image, lower_color, lower_color)

  # 輪郭を検出
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # 輪郭を単純化
  epsilon_factor = 0.02  # 輪郭の単純化度合いの調整
  for contour in contours:
    epsilon = epsilon_factor * cv2.arcLength(contour, True)  # 輪郭の長さに基づいてepsilonを計算
    approx = cv2.approxPolyDP(contour, epsilon, True)  # 輪郭を単純化
    polygon = [(point[0][0], point[0][1]) for point in approx]
    current_polygons.append(polygon)

  # ポリゴンの差分を計算し、赤色で塗りつぶす
  if prev_polygons:
    prev_polygons_union = unary_union([Polygon(p) for p in prev_polygons])
    current_polygons_union = unary_union([Polygon(p) for p in current_polygons])

    # 差分を取得
    difference = current_polygons_union.difference(prev_polygons_union)

    # 差分が MultiPolygon かどうかを確認
    if not difference.is_empty:
      if isinstance(difference, MultiPolygon):
        for poly in difference:
          coords = np.array(list(poly.exterior.coords), dtype=np.int32)
          coords = coords.reshape((-1, 1, 2))
          cv2.fillPoly(current_image, [coords], color=[255, 0, 0])  # 赤色で塗りつぶし
      elif isinstance(difference, Polygon):
        coords = np.array(list(difference.exterior.coords), dtype=np.int32)
        coords = coords.reshape((-1, 1, 2))
        cv2.fillPoly(current_image, [coords], color=[255, 0, 0])  # 赤色で塗りつぶし

      output_image_path = f"step_{step}.png"
      cv2.imwrite(output_image_path, cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))  # BGRに変換して保存
      print(f"ステップ {step} の画像を保存しました: {output_image_path}")
    else:
      breakpoint()

  return current_polygons


def color_regions_step_by_step(image, regions):
  """領域ごとに順番に青色で塗りつぶし、各ステップで画像を保存"""
  colored_image = np.copy(image)  # 画像をコピーして塗りつぶす
  step = 0

  prev_polygons = []
  for region in regions:
    # 各領域を青色 [0, 0, 255] で塗りつぶす
    for (x, y) in region:
      colored_image[x, y] = [0, 0, 255]  # 青色で塗りつぶす (RGB)

    # 各ステップの画像を保存
    current_polygons = simplify_polygons_and_save_image(colored_image, step, prev_polygons)
    prev_polygons = current_polygons
    step += 1

  return colored_image


# メイン処理
def main(image_path):
  # 画像を読み込む (OpenCVはBGRで読み込むのでRGBに変換)
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGBに変換

  # 同じ色の領域に分割
  regions = segment_image_by_color(image)

  # 順番に領域を青色で塗りつぶし、各ステップを保存
  colored_image = color_regions_step_by_step(image, regions)

  # 最終結果の保存
  final_output_image_path = "final_colored_image.png"
  cv2.imwrite(final_output_image_path, cv2.cvtColor(colored_image, cv2.COLOR_RGB2BGR))  # BGRに変換して保存
  print(f"最終結果の画像を保存しました: {final_output_image_path}")


# 実行例
main(image_path)
