import itertools
import math
import os
from pathlib import Path
from collections import deque

import numpy as np
import numpy.typing as npt
from PIL import Image
import cv2
from shapely.geometry import Polygon


class RoofLayerInfo:
  NO_POINT = -1
  NOISE_POINT = -2
  ROOF_LINE_POINT = -3
  ROOF_VERTICE_POINT = -4
  WALL_HEIGHT_THRESHOLD = 0.2

  RESERVED_COLOR = {
      NO_POINT: [255, 255, 255],
      NOISE_POINT: [0, 0, 0],
      ROOF_LINE_POINT: [0, 255, 0],
      ROOF_VERTICE_POINT: [255, 0, 0],
  }

  @property
  def rgb_image(self):
    """
    DSM点群のRGB画像

    Returns:
      npt.NDArray[np.float_]: DSM点群のRGB画像
    """

    return self._rgb_image

  @property
  def layer_class(self):
    """
    DSM点群の画像座標 (i,j) 二次元アレイに壁点を起点としてクラスタリングした屋根のレイヤー番号を記録したもの

    Returns:
      npt.NDArray[np.int_]: DSM点群の画像座標 (i,j) 二次元アレイに壁点を起点としてクラスタリングした屋根のレイヤー番号を記録したもの
    """

    return self._layer_class

  @property
  def layer_number_layer_outline_ijs_list_pair(self):
    """
    クラスタリングされたレイヤー番号に対応するレイヤーのポリゴンリスト

    Returns:
      dict[int, list[list[tuple[int, int]]]]: クラスタリングされたレイヤー番号に対応するレイヤーのポリゴンリスト
    """

    return self._layer_number_layer_outline_ijs_list_pair

  @property
  def wall_point_positions(self):
    """
    壁点リスト

    Returns:
      list[]: 壁点リスト
    """
    return self._wall_point_positions

  @property
  def layer_class_length(self):
    """
    クラスタリングされたレイヤー番号の数

    Returns:
      int: クラスタリングされたレイヤー番号の数
    """
    return self._layer_class_length

  def __init__(
      self,
      layer_points_xyz: npt.NDArray[np.float_],
      rgb_image: npt.NDArray[np.int_],
      debug_dir: str,
      debug_mode: bool = False,
  ):
    """
    屋根線をイメージとして保存（デバッグ用）

    Args:
      layer_points_xyz: DSM点群のRGB画像(i,j)のxyz座標
      rgb_image (npt.NDArray[np.uint8]): DSM点群のRGB画像
      debug_dir (str): 記録するファイル名
      debug_mode (bool): デバッグモード
    """

    self.debug_dir: str = debug_dir
    self._rgb_image = rgb_image.copy()
    self._layer_points_xyz = layer_points_xyz
    self._debug_mode = debug_mode
    self._debug_dir = debug_dir
    self._height, self._width = layer_points_xyz.shape[:2]
    self._layer_class = np.full((self._height, self._width), RoofLayerInfo.NO_POINT, dtype=np.int_)
    self._layer_class_length = 0

    self._color_palette = self.get_color_palette(RoofLayerInfo.RESERVED_COLOR.values())
    self._wall_point_positions = self._get_wall_point_positions(self._layer_points_xyz)
    self._xy_ij = self._get_xy_ij_pair(self._layer_points_xyz)
    self._init_layer_class()
    self._detect_and_mark_noise()

    if self._debug_mode:
      Path(self._debug_dir).mkdir(parents=True, exist_ok=True)

      self._save_image_origin()
      self._save_image_wall_line(self.wall_point_positions)
      self.save_layer_image(self._layer_class)
      self._init_layer_number_outline_ij_pair(self._layer_class)

  def find_nearest_xy(self, search_x: float, search_y: float):
    """
    点群画像の xy座標基準、一番近い点の(x,y)を返す

    Args:
      search_x (float): 選択した任意の点の x
      search_y (float): 選択した任意の点の y
    """

    nearest_xy = None
    min_distance = float('inf')

    for (x, y) in self._xy_ij.keys():
      distance = math.sqrt((search_x - x) ** 2 + (search_y - y) ** 2)
      if distance < min_distance:
        min_distance = distance
        nearest_xy = (x, y)

    return nearest_xy

  def xy_to_ij(self, x, y):
    """
    DSM点群の (x, y) の座標をDSM点群の画像座標である (i, j) へ変換

    Args:
      x (float): 選択した任意の点の x
      y (float): 選択した任意の点の y
    """

    return self._xy_ij[(x, y)]

  def _get_wall_point_positions(self, layer_points_xyz: npt.NDArray[np.float_]):
    """壁の点を設定する"""

    height, width = layer_points_xyz.shape[:2]
    wall_point_positions: list[tuple[float, float]] = []
    for i, layer_points_xyz_j in enumerate(layer_points_xyz):
      for j, (x, y, z1) in enumerate(layer_points_xyz_j):
        if (x == 0 and y == 0 and z1 == 0):
          continue

        z2s = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
          i2, j2 = i + di, j + dj
          if 0 <= i2 < height and 0 <= j2 < width:
            z2s.append(layer_points_xyz[i2, j2, 2])

        if self._is_wall_index(z1, z2s):
          wall_point_positions.append((i, j))

    return wall_point_positions

  def _get_xy_ij_pair(self, layer_points_xyz: npt.NDArray[np.float_]):
    """DSM点群の (x, y) の座標をDSM点群の画像座標である (i, j) へ変換できるようにインデクスを作る"""

    xy_ij: dict[tuple[float, float], tuple[int, int]] = {}
    for i, layer_points_xyz_j in enumerate(layer_points_xyz):
      for j, (x, y, z1) in enumerate(layer_points_xyz_j):
        xy_ij[(x, y)] = (i, j)

    return xy_ij

  def _is_wall_index(self, z1: float, z2s: list[float]):
    """
    壁点を判定する

    Args:
      z1 (float): DSM点群のある点(x, y, z) の z 座標
      z2s (list[float]): DSM点群のある点(x, y, z) の前後左右の点の z 座標（最大4個）
    """
    return any((z1 - z2) > RoofLayerInfo.WALL_HEIGHT_THRESHOLD for z2 in z2s)

  def _bfs_layer_fill(self, start_i: int, start_j: int):
    """
    BFS を使ってレイヤーの点を探索し、layer_class を更新

    Args:
      start_i (int): DSM点群のRGB画像の位置(i,j) の i
      start_j (int): DSM点群のRGB画像の位置(i,j) の j
    """

    queue = deque([(start_i, start_j)])  # BFS のためのキュー
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右の方向

    # 初期地点の z 座標
    self._layer_class[start_i, start_j] = self._layer_class_length

    while queue:
      i, j = queue.popleft()

      # 現在の点の z 座標
      current_z = self._layer_points_xyz[i, j, 2]

      # 前後左右の点を探索
      for di, dj in directions:
        i2, j2 = i + di, j + dj

        # 境界チェック
        if 0 <= i2 < self._height and 0 <= j2 < self._width:
          if self._layer_class[i2, j2] == RoofLayerInfo.NO_POINT:
            neighbor_z = self._layer_points_xyz[i2, j2, 2]

            # z 座標の差が RoofLayerInfo.WALL_HEIGHT_THRESHOLD 以下なら、同じレイヤーと見なす
            if abs(current_z - neighbor_z) <= RoofLayerInfo.WALL_HEIGHT_THRESHOLD:
              self._layer_class[i2, j2] = self._layer_class_length
              queue.append((i2, j2))  # 探索対象としてキューに追加

  def _init_layer_class(self):
    """wall_point_positions を基にレイヤーを割り当てる処理"""

    # wall_point_positions から BFS を開始して各レイヤーを探索
    for i, j in self._wall_point_positions:
      if self._layer_class[i, j] == RoofLayerInfo.NO_POINT:
        self._bfs_layer_fill(i, j)
        if self._layer_class_length > 47:
          continue
        self._layer_class_length += 1  # 次のレイヤー番号に進む

  def _detect_and_mark_noise(self):
    """すべての壁領域クラスをループし、ノイズを検出してマークする"""
    for layer_number in range(self._layer_class_length):
      # 現在の layer_number に属する (i, j) のリストを収集
      layer_points = [(i, j) for i in range(self._height) for j in range(self._width)
                      if self._layer_class[i, j] == layer_number]

      if not layer_points:
        continue  # そのクラスに点がない場合はスキップ

      has_noise = True
      # ノイズの場合、そのクラス全体の点を RoofLayerInfo.NOISE_POINT にマーク
      for i, j in layer_points:
        if self._is_ok_point(i, j, layer_number):
          has_noise = False
          break

      if has_noise:
        for i, j in layer_points:
          self._layer_class[i, j] = RoofLayerInfo.NOISE_POINT

  def _is_ok_point(self, start_i: int, start_j: int, layer_number: int):
    """
    指定されたクラスの点を探索し、ノイズかどうかをチェックする

    Args:
      start_i (int): DSM点群のRGB画像の位置(i,j) の i
      start_j (int): DSM点群のRGB画像の位置(i,j) の j
      layer_number (int): 壁点を起点としてクラスタリングした屋根のレイヤー番号
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右の方向

    ok_count = 0
    for di, dj in directions:
      i2, j2 = start_i + di, start_j + dj

      # 境界チェック
      if 0 <= i2 < self._height and 0 <= j2 < self._width:
        if self._layer_class[i2, j2] == layer_number:
          ok_count += 1
      else:
        ok_count += 1

    return ok_count == 4

  def _save_image_origin(self):
    """
    DSM点群のRGB画像原本をイメージとして保存（デバッグ用）
    """
    image_origin = Image.fromarray(self._rgb_image, "RGB")
    image_origin_path = os.path.join(self._debug_dir, 'origin.png')
    image_origin.save(image_origin_path)

  def _save_image_wall_line(
      self,
      wall_point_positions: list[tuple[int, int]],
      file_name: str = 'wall_line.png',
  ):
    """
    壁の点をイメージとして保存（デバッグ用）

    Args:
      wall_point_positions (list[tuple[int, int]]): DSM点群のRGB画像で壁の点の位置(i, j)
      file_name (str): 記録するファイル名
    """
    rgb_image_wall_line = self._rgb_image.copy()
    for i, j in wall_point_positions:
      rgb_image_wall_line[i, j] = [255, 0, 0]

    image_wall_line = Image.fromarray(rgb_image_wall_line, "RGB")
    image_wall_line_path = os.path.join(self._debug_dir, file_name)
    image_wall_line.save(image_wall_line_path)

  def save_layer_image(self, layer_class: npt.NDArray[np.int_], file_name: str = 'layer.png'):
    """
    屋根レイヤーをイメージとして保存（デバッグ用）

    Args:
      layer_class (npt.NDArray[np.int_]): DSM点群の画像座標 (i,j) 二次元アレイに壁点を起点としてクラスタリングした屋根のレイヤー番号を記録したもの
      file_name (str): 記録するファイル名
    """
    height, width = layer_class.shape

    # 空の RGB 画像を作成 (すべて白で初期化)
    image_rgb = np.full((height, width, 3), 255, dtype=np.uint8)

    # i, j に基づいて各ピクセルに色を割り当て
    for i in range(height):
      for j in range(width):
        layer_number = layer_class[i, j]
        image_rgb[i, j] = self.get_color(layer_number)  # レイヤーに対応する色を設定

    image_layer = Image.fromarray(image_rgb, 'RGB')
    image_layer_path = os.path.join(self._debug_dir, file_name)
    image_layer.save(image_layer_path)

  def save_roof_line_image(
      self,
      rgb_image: npt.NDArray[np.uint8],
      roof_lines: set[tuple[tuple[int, int], tuple[int, int]]],
      file_name: str = 'roof_line.png',
  ):
    """
    屋根線をイメージとして保存（デバッグ用）

    Args:
      rgb_image (npt.NDArray[np.uint8]): DSM点群のRGB画像
      roof_lines (set[tuple[tuple[int, int], tuple[int, int]]]): 線のリスト
      file_name (str): 記録するファイル名
    """

    # RGB画像データをBGRに変換（OpenCVはBGRを使用）
    image_roof_line = cv2.cvtColor(rgb_image.copy(), cv2.COLOR_RGB2BGR)

    # 線と点を一緒に描画
    for start_image_pos, end_image_pos in roof_lines:
        # 座標を整数にキャストして x, y を入れ替え
      start_image_pos = tuple(map(int, (start_image_pos[1], start_image_pos[0])))
      end_image_pos = tuple(map(int, (end_image_pos[1], end_image_pos[0])))

      # 緑色の線、太さ1
      cv2.line(
          image_roof_line,
          start_image_pos,
          end_image_pos,
          self.get_color(RoofLayerInfo.ROOF_LINE_POINT),  # 線の色
          1
      )

      # 赤色の点を描画
      cv2.circle(
          image_roof_line,
          start_image_pos,
          0,  # 点の半径
          self.get_color(RoofLayerInfo.ROOF_VERTICE_POINT),  # 赤色
          -1  # 塗りつぶし
      )

      cv2.circle(
          image_roof_line,
          end_image_pos,
          0,  # 点の半径
          self.get_color(RoofLayerInfo.ROOF_VERTICE_POINT),  # 赤色
          -1  # 塗りつぶし
      )

    # ファイルに保存
    image_roof_line_path = os.path.join(self._debug_dir, file_name)

    cv2.imwrite(image_roof_line_path, image_roof_line)

  def get_color_palette(self, reserved_colors: list[list[int]]):
    color_palette_all: list[tuple[int, int, int]] = []
    for color_variation_num in [255, 128, 64, 32, 16, 8]:
      # 0, 255 の組み合わせの色を先に出す
      # 次は 0, 128, 255 の組み合わせの色を先に出す
      base_colors = list(range(0, 256, color_variation_num))
      color_palette = list(itertools.product(base_colors, repeat=3))  # すべての組み合わせを生成
      color_palette_all += list(set(color_palette) - set(color_palette_all))

    # 予約された色は抜いておく
    return [
        list(color_palette) for color_palette in color_palette_all
        if (list(color_palette) not in reserved_colors)
    ]

  def get_color(self, color_index: int):
    # 4096 の色を作る
    if color_index == RoofLayerInfo.NO_POINT:
      return [255, 255, 255]

    if color_index == RoofLayerInfo.NOISE_POINT:
      return [0, 0, 0]

    if color_index == RoofLayerInfo.ROOF_LINE_POINT:
      return [0, 255, 0]

    return self._color_palette[color_index]

  def _init_layer_number_outline_ij_pair(self, layer_class: np.ndarray, file_name: str = 'layer_outline.png'):
    """
    レイヤーの外形線を保存する

    Args:
      layer_class (np.ndarray): DSM点群の画像座標 (i,j) 二次元アレイに壁点を起点としてクラスタリングした屋根のレイヤー番号を記録したもの
      file_name (str): 記録するファイル名
    """

    self._layer_number_layer_outline_ijs_list_pair: dict[int, list[list[tuple[int, int]]]] = {}

    height, width = layer_class.shape

    # 空の RGB 画像を作成 (すべて白で初期化)
    image_rgb = np.full((height, width, 3), 255, dtype=np.uint8)
    outline_image_rgb = np.full((height, width, 3), 255, dtype=np.uint8)

    layer_numbers: set[int] = set()

    # i, j に基づいて各ピクセルに色を割り当て
    for i in range(height):
      for j in range(width):
        layer_number = layer_class[i, j]
        layer_numbers.add(layer_number)
        image_rgb[i, j] = self.get_color(layer_number)  # レイヤーに対応する色を設定

    for layer_number in layer_numbers:
      if layer_number < 0:
        continue

      rgb_color = self.get_color(layer_number)
      mask = cv2.inRange(image_rgb, np.array(rgb_color), np.array(rgb_color))
      contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      # 輪郭を単純化
      simplified_contours = []
      layer_outline_ijs: list[list[tuple[int, int]]] = []
      epsilon_factor = 0.02  # 輪郭の単純化度合いの調整 (この値を調整するとポリゴンの単純化が変わる)
      for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)  # 輪郭の長さに基づいてepsilonを計算
        approx = cv2.approxPolyDP(contour, epsilon, True)  # 輪郭を単純化
        layer_outline_ij = [(point[0][0], point[0][1]) for point in approx]

        # 頂点が二つ以下の場合はポリゴンではない
        if len(layer_outline_ij) >= 3 and Polygon(layer_outline_ij).is_valid:
          simplified_contours.append(approx)
          layer_outline_ijs.append(layer_outline_ij)

      self._layer_number_layer_outline_ijs_list_pair[layer_number] = layer_outline_ijs

      # 輪郭線を描画 (塗りつぶさない)
      cv2.polylines(outline_image_rgb, simplified_contours, isClosed=True, color=rgb_color, thickness=1)

    if self._debug_mode:
      outline_image_rgb = cv2.cvtColor(outline_image_rgb, cv2.COLOR_BGR2RGB)
      image_layer_path = os.path.join(self._debug_dir, file_name)
      cv2.imwrite(image_layer_path, outline_image_rgb)
