import os
from pathlib import Path
from typing import Final, Optional
import math

import numpy as np
import numpy.typing as npt
from PIL import Image
import shapely.geometry as geo
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors

from .coordinates_converter import CoordConverterForCartesianAndImagePos
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
      wall_height_threshold: float = 0.5,
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

  def preprocess(
      self,
      cloud: PointCloud,
      ground_height: float,
      footprint: geo.Polygon,
      debug_mode: bool = False
  ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
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
    """

    wall_points = []

    if False:
      points = cloud.get_points().copy()
      colors = cloud.get_colors().copy()

      # 屋根線検出、バルコニー検出用の画像を作成

      min_x, min_y = points[:, :2].min(axis=0)
      max_x, max_y = points[:, :2].max(axis=0)

      height = round((max_y - min_y) / self._grid_size) + 1
      width = round((max_x - min_x) / self._grid_size) + 1

      coordinates_converter = CoordConverterForCartesianAndImagePos(
          grid_size=self._grid_size,
          cartesian_coord_upper_left=(min_x, max_y)
      )

      # 画像にする
      rgb_image = np.full((height, width, 3), 255, dtype=np.uint8)
      depth_image = np.full((height, width), 255, dtype=np.uint8)

      for (x, y, z), rgb in zip(points, colors):
        px_x, px_y = coordinates_converter.cartesian_point_to_image_point(x, y)
        rgb_image[px_y][px_x] = rgb / 256.0
        lower, upper = ground_height - 5, ground_height + 25
        depth_image[px_y][px_x] = (
            np.clip(z, lower, upper) - lower) / (upper - lower) * 255

    else:
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

      wall_xyz = pc_xyz[inds]
      rgb_image = pc_rgb[inds] / 256

      lower, upper = ground_height - 5, ground_height + 25
      depth_image = (np.clip(pc_xyz[:, 2][inds], lower, upper) - lower) / (upper - lower) * 255

      for i, xy_ in enumerate(xy):
        p = Point(xy_[0], xy_[1])
        if not footprint.contains(p):
          wall_xyz[i] = 0
          rgb_image[i] = 255
          depth_image[i] = 255

      wall_xyz = wall_xyz.reshape(height, width, 3).astype(np.float_)
      rgb_image = rgb_image.reshape(height, width, 3).astype(np.uint8)
      depth_image = depth_image.reshape(height, width).astype(np.uint8)

      if debug_mode:
        rgb_image_wall_line = rgb_image.copy()
        for i, wall_xyz_j in enumerate(wall_xyz):
          for j, (x, y, z1) in enumerate(wall_xyz_j):
            if (x == 0 and y == 0 and z1 == 0):
              continue

            z2s = []
            try:
              z2s.append(wall_xyz[i - 1, j][2])
            except Exception: pass

            try:
              z2s.append(wall_xyz[i + 1, j][2])
            except Exception: pass

            try:
              z2s.append(wall_xyz[i, j - 1][2])
            except Exception: pass

            try:
              z2s.append(wall_xyz[i, j + 1][2])
            except Exception: pass

            if self._is_wall_point(z1, z2s):
              wall_points.append([x, y, z1])
              rgb_image_wall_line[i][j] = [255, 0, 0]

        Path(os.path.join('debug', self._building_id)).mkdir(parents=True, exist_ok=True)
        image_origin = Image.fromarray(rgb_image, "RGB")
        image_wall_line = Image.fromarray(rgb_image_wall_line, "RGB")

        image_origin_path = os.path.join('debug', self._building_id, 'origin.jpg')
        image_wall_line_path = os.path.join('debug', self._building_id, 'wall_line.jpg')

        image_origin.save(image_origin_path)
        image_wall_line.save(image_wall_line_path)

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

    return square_rgb_image, square_depth_image, wall_points

  def _is_wall_point(self, z1: float, z2s: list[float]):
    """
    Check DSM point is wall point
    """
    for z2 in z2s:
      # mark only high point as wall point
      if (z1 - z2) > self._wall_height_threshold: return True

    return False
