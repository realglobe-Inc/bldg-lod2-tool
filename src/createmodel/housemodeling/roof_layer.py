from collections import deque
import numpy as np
import numpy.typing as npt


class RoofLayer:
  NO_POINT = -1
  NOISE_POINT = -2
  WALL_HEIGHT_THRESHOLD = 0.2

  @property
  def layer_class(self):
    return self._layer_class

  @property
  def wall_indexes(self):
    return self._wall_indexes

  def __init__(self, layer_point_xyz: npt.NDArray[np.float_]):
    self._layer_point_xyz = layer_point_xyz
    self._height, self._width = layer_point_xyz.shape[:2]
    self._layer_class = np.full((self._height, self._width), RoofLayer.NO_POINT, dtype=np.int_)
    self._layer_count = 0
    self._wall_indexes = []
    self._set_wall_indexes()
    self._set_layer_class()
    self._detect_and_mark_noise()

  def _set_wall_indexes(self):
    for i, layer_point_xyz_j in enumerate(self._layer_point_xyz):
      for j, (x, y, z1) in enumerate(layer_point_xyz_j):
        if (x == 0 and y == 0 and z1 == 0):
          continue
        z2s = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
          i2, j2 = i + di, j + dj
          if 0 <= i2 < self._height and 0 <= j2 < self._width:
            z2s.append(self._layer_point_xyz[i2, j2, 2])
        if self._is_wall_index(z1, z2s):
          self._wall_indexes.append((i, j))

  def _is_wall_index(self, z1: float, z2s: list[float]):
    """Check if DSM point is a wall point"""
    return any((z1 - z2) > RoofLayer.WALL_HEIGHT_THRESHOLD for z2 in z2s)

  def _bfs_layer_fill(self, start_i: int, start_j: int):
    """BFS を使ってレイヤーの点を探索し、layer_class を更新"""
    queue = deque([(start_i, start_j)])  # BFS のためのキュー
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右の方向

    # 初期地点の z 座標
    self._layer_class[start_i, start_j] = self._layer_count

    while queue:
      i, j = queue.popleft()

      # 現在の点の z 座標
      current_z = self._layer_point_xyz[i, j, 2]

      # 前後左右の点を探索
      for di, dj in directions:
        i2, j2 = i + di, j + dj

        # 境界チェック
        if 0 <= i2 < self._height and 0 <= j2 < self._width:
          # すでに self._layer_count が設定されていないか確認
          if self._layer_class[i2, j2] == RoofLayer.NO_POINT:
            neighbor_z = self._layer_point_xyz[i2, j2, 2]

            # z 座標の差が RoofLayer.WALL_HEIGHT_THRESHOLD 以下なら、同じレイヤーと見なす
            if abs(current_z - neighbor_z) <= RoofLayer.WALL_HEIGHT_THRESHOLD:
              self._layer_class[i2, j2] = self._layer_count
              queue.append((i2, j2))  # 探索対象としてキューに追加

  def _set_layer_class(self):
    """wall_indexes を基にレイヤーを割り当てる処理"""
    visited = set()  # すでに探索済みの場所を記録

    # wall_indexes から BFS を開始して各レイヤーを探索
    for i, j in self._wall_indexes:
      if (i, j) not in visited and self._layer_class[i, j] == RoofLayer.NO_POINT:
        self._bfs_layer_fill(i, j)
        visited.add((i, j))
        self._layer_count += 1  # 次のレイヤー番号に進む

  def _detect_and_mark_noise(self):
    """すべての壁領域クラスをループし、ノイズを検出してマークする"""
    for layer_number in range(self._layer_count):
      # 現在の layer_number に属する (i, j) のリストを収集
      layer_points = [(i, j) for i in range(self._height) for j in range(self._width)
                      if self._layer_class[i, j] == layer_number]

      if not layer_points:
        continue  # そのクラスに点がない場合はスキップ

      has_noise = True
      # ノイズの場合、そのクラス全体の点を RoofLayer.NOISE_POINT にマーク
      for i, j in layer_points:
        if self._is_ok_point(i, j, layer_number):
          has_noise = False
          break

      if has_noise:
        for i, j in layer_points:
          self._layer_class[i, j] = RoofLayer.NOISE_POINT

  def _is_ok_point(self, start_i: int, start_j: int, layer_number: int):
    """指定されたクラスの点を探索し、ノイズかどうかをチェックする"""
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
