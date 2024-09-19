from collections import deque
import os
from pathlib import Path
import random
from typing import Final, Optional
import math

import numpy as np
import numpy.typing as npt
from PIL import Image
import shapely.geometry as geo
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors

from ..lasmanager import PointCloud


class Preprocess:
  """前処理クラス
  """

  _grid_size: Final[float]
  _image_size: Final[int]
  _expand_rate: Final[float]

  def __init__(
      self,
      grid_size: float,
      image_size: int,
      expand_rate: Optional[float] = None,
      wall_height_threshold: float = 0.2,
      building_id: Optional[str] = None,
  ) -> None:
    """コンストラクタ

    Args:
        grid_size(float): 点群の間隔(meter)
        image_size(int): 出力する画像のサイズ(pixel)
        expand_rate(float, optional): 画像の拡大率 (Default: 1)
    """

    self._grid_size = grid_size
    self._image_size = image_size
    self._expand_rate = expand_rate if expand_rate is not None else 1.0
    self._wall_height_threshold = wall_height_threshold
    self._building_id = building_id

    """コンストラクタ
      """
    self._XYZ = 'xyz'
    self._RGB = 'rgb'
    self._IND = 'ind'

  def preprocess(
      self,
      cloud: PointCloud,
      ground_height: float,
      footprint: geo.Polygon,
      debug_mode: bool = False
  ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], list]:
    """前処理

    点群から機械学習モデルへの入力用の画像を作成する

    Args:
      cloud(PointCloud): 建物点群
      ground_height (float): 地面の高さm
      footprint(geo.Polygon): 建物外形ポリゴン
      debug_mode (bool, optional): デバッグモード (Default: False)

    Returns:
      NDArray[np.uint8]: (image_size, image_size, 3)のRGB画像データ
      NDArray[np.uint8]: (image_size, image_size)の高さのグレースケール画像データ
      list: 検出された壁点
    """

    # 屋根線検出、バルコニー検出用の画像を作成
    pc_xyz = cloud.get_points().copy()
    pc_rgb = cloud.get_colors().copy()

    pc_x_min, pc_y_min, _ = pc_xyz.min(axis=0)
    pc_x_max, pc_y_max, _ = pc_xyz.max(axis=0)

    width = math.ceil((pc_x_max - pc_x_min) / self._grid_size) + 1
    height = math.ceil((pc_y_max - pc_y_min) / self._grid_size) + 1

    xs = np.arange(width) * self._grid_size + pc_x_min
    ys = -np.arange(height) * self._grid_size + pc_y_max
    xx, yy = np.meshgrid(xs, ys)
    xy = np.dstack([xx, yy]).reshape(-1, 2)

    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', leaf_size=30, n_jobs=4)
    nn.fit(pc_xyz[:, 0:2])
    inds = nn.kneighbors(xy, return_distance=False)[:, 0]

    layer_point_xyz = pc_xyz[inds]
    rgb_image = pc_rgb[inds] / 256

    lower, upper = ground_height - 5, ground_height + 25
    depth_image = (np.clip(pc_xyz[:, 2][inds], lower, upper) - lower) / (upper - lower) * 255

    for i, xy_ in enumerate(xy):
      p = Point(xy_[0], xy_[1])
      if not footprint.contains(p):
        layer_point_xyz[i] = 0
        rgb_image[i] = 255
        depth_image[i] = 255

    layer_point_xyz = layer_point_xyz.reshape(height, width, 3).astype(np.float_)
    rgb_image = rgb_image.reshape(height, width, 3).astype(np.uint8)
    depth_image = depth_image.reshape(height, width).astype(np.uint8)

    layer_indexes = np.full((height, width), -1, dtype=np.int_)

    if debug_mode:
      wall_indexes = self._get_wall_indexes(layer_point_xyz)
      layer_indexes, layer_count = self._assign_layers(wall_indexes, layer_point_xyz)

      self._save_layer_image(layer_indexes, layer_count)
      self._save_image_origin(rgb_image)
      self._save_image_wall_line(rgb_image, wall_indexes)

    # 画像を拡大
    if self._expand_rate != 1:
      expanded_size = (round(width * self._expand_rate), round(height * self._expand_rate))
      rgb_image = np.array(Image.fromarray(rgb_image).resize(expanded_size), dtype=np.uint8)
      depth_image = np.array(
          Image.fromarray(depth_image, 'L').resize(expanded_size), dtype=np.uint8
      )

      width, height = expanded_size

    # モデル入力用の正方形画像に変換(余白は白で埋める)
    square_rgb_image = np.full((self._image_size, self._image_size, 3), 255, dtype=np.uint8)
    square_depth_image = np.full((self._image_size, self._image_size), 255, dtype=np.uint8)

    top = (self._image_size - height) // 2
    left = (self._image_size - width) // 2

    square_rgb_image[top:top + height, left:left + width] = rgb_image
    square_depth_image[top:top + height, left:left + width] = depth_image

    return square_rgb_image, square_depth_image

  def _is_wall_indexe(self, z1: float, z2s: list[float]):
    """
    Check DSM point is wall point
    """
    for z2 in z2s:
      # mark only high point as wall point
      if (z1 - z2) > self._wall_height_threshold: return True

    return False

  def _save_image_origin(self, rgb_image: npt.NDArray[np.float_]):
    image_origin = Image.fromarray(rgb_image, "RGB")
    image_origin_path = os.path.join('debug', self._building_id, 'origin.jpg')
    image_origin.save(image_origin_path)

  def _save_image_wall_line(
      self,
      rgb_image: npt.NDArray[np.uint8],
      wall_indexes: list[tuple[int, int]]
  ):
    rgb_image_wall_line = rgb_image.copy()
    for i, j in wall_indexes:
      rgb_image_wall_line[i, j] = [255, 0, 0]

    image_wall_line = Image.fromarray(rgb_image_wall_line, "RGB")

    Path(os.path.join('debug', self._building_id)).mkdir(parents=True, exist_ok=True)
    image_wall_line_path = os.path.join('debug', self._building_id, 'wall_line.jpg')
    image_wall_line.save(image_wall_line_path)

  def _save_layer_image(self, layer_indexes: npt.NDArray[np.int_], layer_count: int):
    """レイヤーごとの色を割り当てて画像を作成する"""
    height, width = layer_indexes.shape

    # 空の RGB 画像を作成 (すべて白で初期化)
    image_rgb = np.full((height, width, 3), 255, dtype=np.uint8)

    # i, j に基づいて各ピクセルに色を割り当て
    for i in range(height):
      for j in range(width):
        layer = layer_indexes[i, j]
        image_rgb[i, j] = self._get_color(layer)  # レイヤーに対応する色を設定

    image_layer = Image.fromarray(image_rgb, 'RGB')

    Path(os.path.join('debug', self._building_id)).mkdir(parents=True, exist_ok=True)
    image_layer_path = os.path.join('debug', self._building_id, 'layer.png')
    image_layer.save(image_layer_path)

  def _get_wall_indexes(self, layer_point_xyz: npt.NDArray[np.float_]):
    wall_indexes: list[tuple[int, int]] = []

    for i, layer_point_xyz_j in enumerate(layer_point_xyz):
      for j, (x, y, z1) in enumerate(layer_point_xyz_j):
        if (x == 0 and y == 0 and z1 == 0):
          continue

        z2s = []
        try:
          z2s.append(layer_point_xyz[i - 1, j][2])
        except Exception: pass

        try:
          z2s.append(layer_point_xyz[i + 1, j][2])
        except Exception: pass

        try:
          z2s.append(layer_point_xyz[i, j - 1][2])
        except Exception: pass

        try:
          z2s.append(layer_point_xyz[i, j + 1][2])
        except Exception: pass

        if self._is_wall_indexe(z1, z2s):
          wall_indexes.append((i, j))

    return wall_indexes

  def _bfs_layer_fill(self, start_i, start_j, layer_count, layer_indexes, layer_point_xyz):
    """BFS を使ってレイヤーの点を探索し、layer_indexes を更新"""
    height, width = layer_indexes.shape
    queue = deque([(start_i, start_j)])  # BFS のためのキュー
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右の方向

    # 初期地点の z 座標
    layer_indexes[start_i, start_j] = layer_count

    while queue:
      i, j = queue.popleft()

      # 現在の点の z 座標
      current_z = layer_point_xyz[i, j, 2]

      # 前後左右の点を探索
      for di, dj in directions:
        i2, j2 = i + di, j + dj

        # 境界チェック
        if 0 <= i2 < height and 0 <= j2 < width:
          # すでに layer_count が設定されていないか確認
          if layer_indexes[i2, j2] == -1:
            neighbor_z = layer_point_xyz[i2, j2, 2]

            # z 座標の差が self._wall_height_threshold 以下なら、同じレイヤーと見なす
            if abs(current_z - neighbor_z) <= self._wall_height_threshold:
              layer_indexes[i2, j2] = layer_count
              queue.append((i2, j2))  # 探索対象としてキューに追加

  def _assign_layers(self, wall_indexes, layer_point_xyz):
    """wall_indexes を基にレイヤーを割り当てる処理"""
    height, width = layer_point_xyz.shape[:2]
    layer_indexes = np.full((height, width), -1, dtype=np.int_)  # 初期化（-1）

    layer_count = 0  # レイヤー番号

    # wall_indexes から BFS を開始して各レイヤーを探索
    for i, j in wall_indexes:
      if layer_indexes[i, j] == -1:  # 未探索の場所なら
        self._bfs_layer_fill(i, j, layer_count, layer_indexes, layer_point_xyz)
        layer_count += 1  # 次のレイヤー番号に進む

    return layer_indexes, layer_count

  def _get_color(self, color_index: int):
    color_palette = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
        [0, 255, 255], [128, 0, 0], [128, 128, 0], [0, 128, 0], [128, 0, 128],
        [0, 128, 128], [0, 0, 128], [255, 165, 0], [255, 20, 147], [75, 0, 130],
        [255, 69, 0], [154, 205, 50], [72, 61, 139], [199, 21, 133], [0, 100, 0],
        [127, 255, 212], [30, 144, 255], [255, 182, 193], [100, 149, 237], [0, 255, 127],
        [255, 105, 180], [147, 112, 219], [60, 179, 113], [218, 112, 214], [220, 20, 60],
        [144, 238, 144], [139, 69, 19], [255, 228, 181], [34, 139, 34], [173, 255, 47],
        [255, 140, 0], [46, 139, 87], [50, 205, 50], [0, 191, 255], [123, 104, 238],
        [255, 228, 225], [245, 222, 179], [139, 0, 0], [205, 133, 63], [255, 218, 185],
        [70, 130, 180], [250, 128, 114], [176, 224, 230], [127, 255, 0], [102, 205, 170]
    ]

    if color_index == -1:
      return [255, 255, 255]

    if color_index < 50:
      return color_palette[color_index]

    return [random.randint(0, 255) for _ in range(3)]
