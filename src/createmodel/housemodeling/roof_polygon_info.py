import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point as GeoPoint

from .model_surface_creation.utils.geometry import Point
from .roof_layer_info import RoofLayerInfo


class RoofPolygonInfo:
  def __init__(
      self,
      cartesian_points: list[Point],
      inner_polygons: list[list[int]],
      roof_layer_info: RoofLayerInfo,
      debug_mode: bool = False
  ):
    """
    屋根ポリゴン情報を管理するクラス

    Args:
      cartesian_points (list[Point]): 各頂点の3D座標リスト（点群データ）
      inner_polygons (list[list[int]]): 屋根の内部ポリゴン（開口部など）を構成する各頂点のインデックスのリストのリスト
      roof_layer_info (RoofLayerInfo): DSM点群に基づくレイヤー情報を管理するクラス
      debug_mode (bool): デバッグモードのフラグ（デフォルトはFalse）
    """
    self._cartesian_points = cartesian_points
    self._inner_polygons = inner_polygons
    self._roof_layer_info = roof_layer_info
    self._debug_mode = debug_mode
    self._height, self._width = self._roof_layer_info.layer_class.shape
    self._layer_class_origin = self._roof_layer_info.layer_class.copy()

    # Cartesian座標と対応する画像座標の対応付け
    self._image_positions: list[Point] = self._calculate_image_positions()

    # RGBイメージの初期化
    self._rgb_image_of_roof_line_with_layer_class_as_is = np.full((self._height, self._width, 3), 255, dtype=np.uint8)
    self._rgb_image_of_roof_line_with_layer_class_to_be = np.full((self._height, self._width, 3), 255, dtype=np.uint8)
    self._rgb_image_of_roof_line_with_layer_class = np.full((self._height, self._width, 3), 255, dtype=np.uint8)

    self._polygon_edges_for_debug_image = set()
    self.has_too_much_noise_on_dsm = False
    self._process_polygons()

    for inner_polygon in self._inner_polygons:
      if self._is_devidable_poligon(inner_polygon):
        start_point_id_end_point_ijs_pair = self._get_devidable_point_on_poligon(inner_polygon)

  def _calculate_image_positions(self):
    """
    Cartesian座標 (x, y) を画像座標 (i, j) に変換し、2次元リストに保存する。

    Returns:
      list[list[int]]: Cartesian座標に対応する画像座標(i, j)のリスト
    """
    image_positions: list[Point] = []
    for point in self._cartesian_points:
      nearest_x, nearest_y = self._roof_layer_info.find_nearest_xy(point.x, point.y)
      nearest_i, nearest_j = self._roof_layer_info.xy_to_ij(nearest_x, nearest_y)
      image_positions.append(Point(nearest_i, nearest_j))

    return image_positions

  def _get_polygon_ij(self, inner_polygon: list[int]):
    """
    ポリゴンの頂点を(i, j)座標に変換する。

    Args:
      inner_polygon (list[int]): ポリゴンを構成する頂点のインデックスリスト

    Returns:
      list[Point]: 各頂点の(i, j)座標のリスト
    """
    return [
        (self._image_positions[point_id].x, self._image_positions[point_id].y)
        for point_id in inner_polygon
    ]

  def _process_polygons(self):
    """
    内部ポリゴンの処理を行い、各種画像を生成する。
    """
    for inner_polygon in self._inner_polygons:
      self._update_polygon_edges_for_debug_image(inner_polygon)
      self._process_polygon_layers(inner_polygon)

    if self._debug_mode:
      self._save_debug_images()

  def _update_polygon_edges_for_debug_image(self, inner_polygon: list[int]):
    """
    ポリゴンの外周線を取得し、デバッグ用のエッジリストに追加する。

    Args:
      inner_polygon (list[int]): ポリゴンを構成する頂点のインデックスリスト
    """
    inner_polygon_ij = self._get_polygon_ij(inner_polygon)
    poly = Polygon(inner_polygon_ij)
    coords = list(poly.exterior.coords)  # ポリゴンの外周座標リスト
    for i in range(len(coords) - 1):  # -1 は最後のエッジを無視するため
      sorted_coord = tuple(sorted([coords[i], coords[i + 1]]))  # 重複を防ぐため、sort
      self._polygon_edges_for_debug_image.add(sorted_coord)

  def _process_polygon_layers(self, inner_polygon: list[int]):
    """
    ポリゴン内のレイヤー情報を処理し、ノイズを判定する。

    Args:
      inner_polygon (list[int]): ポリゴンを構成する頂点のインデックスリスト
    """
    layer_number_points_pair = self._get_layer_number_points_pair(inner_polygon)
    majority_layer_number = self._get_majority_layer_number(layer_number_points_pair)

    if majority_layer_number < 0:
      self.has_too_much_noise_on_dsm = True

    self._update_rgb_images(layer_number_points_pair, majority_layer_number)

  def _get_layer_number_points_pair(self, inner_polygon: list[int]):
    """
    ポリゴン内のレイヤー番号と対応するポイントを取得する。

    Args:
        inner_polygon (list[int]): ポリゴンを構成する頂点のインデックスリスト

    Returns:
        dict[int, list[tuple[int, int]]]: レイヤー番号とそのポイント(i, j)のペアを含む辞書
    """
    inner_polygon_ij = self._get_polygon_ij(inner_polygon)
    poly = Polygon(inner_polygon_ij)
    layer_number_points_pair: dict[int, list[tuple[int, int]]] = {}
    for i in range(self._height):
      for j in range(self._width):
        is_inside_polygon = poly.contains(GeoPoint(i, j))
        if is_inside_polygon:
          layer_number = self._layer_class_origin[i, j]

          if layer_number_points_pair.get(layer_number) is None:
            layer_number_points_pair[layer_number] = []

          layer_number_points_pair[layer_number].append((i, j))
    return layer_number_points_pair

  def _get_majority_layer_number(self, layer_number_points_pair: dict[int, list[tuple[int, int]]]):
    """
    ポリゴン内で最も多く出現するレイヤー番号を取得する。

    Args:
      layer_number_points_pair (dict[int, list[tuple[int, int]]]): レイヤー番号とそのポイント(i, j)のペアを含む辞書

    Returns:
      int: ポリゴン内で最も多く出現するレイヤー番号
    """
    layer_count_max = 0
    majority_layer_number = RoofLayerInfo.NOISE_POINT
    for layer_number, layer_points_ij in layer_number_points_pair.items():
      layer_count = len(layer_points_ij)

      if layer_count > layer_count_max and layer_number != RoofLayerInfo.NOISE_POINT:
        layer_count_max = layer_count
        majority_layer_number = layer_number

    return majority_layer_number

  def _update_rgb_images(
      self,
      layer_number_points_pair: dict[int, list[tuple[int, int]]],
      majority_layer_number: int,
  ):
    """
    RGBイメージを更新する。

    Args:
      layer_number_points_pair (dict[int, list[tuple[int, int]]]): レイヤー番号とそのポイント(i, j)のペアを含む辞書
      majority_layer_number (int): ポリゴン内の多数派のレイヤー番号
    """
    for layer_number, layer_points_ij in layer_number_points_pair.items():
      for i, j in layer_points_ij:
        self._rgb_image_of_roof_line_with_layer_class_as_is[i, j] = self._roof_layer_info._get_color(layer_number)
        self._rgb_image_of_roof_line_with_layer_class_to_be[i, j] = self._roof_layer_info._get_color(majority_layer_number)
        self._rgb_image_of_roof_line_with_layer_class[i, j] = self._roof_layer_info._get_color(
            RoofLayerInfo.NOISE_POINT if layer_number != layer_number else self._layer_class_origin[i, j]
        )

  def _save_debug_images(self):
    """
    デバッグ用の画像を保存する。
    """
    self._roof_layer_info.save_roof_line_image(
        self._roof_layer_info.rgb_image,
        self._polygon_edges_for_debug_image
    )
    self._roof_layer_info.save_roof_line_image(
        self._rgb_image_of_roof_line_with_layer_class_as_is,
        self._polygon_edges_for_debug_image,
        'roof_line_with_layer_class_as_is.png',
    )
    self._roof_layer_info.save_roof_line_image(
        self._rgb_image_of_roof_line_with_layer_class_to_be,
        self._polygon_edges_for_debug_image,
        'roof_line_with_layer_class_to_be.png',
    )
    self._roof_layer_info.save_roof_line_image(
        self._rgb_image_of_roof_line_with_layer_class,
        self._polygon_edges_for_debug_image,
        'roof_line_with_layer_class.png',
    )

  def _ensure_counterclockwise(self, inner_polygon: list[int]):
    """
    ポリゴンの頂点が反時計回りになるようにする
    Args:
      inner_polygon (list[int]): ポリゴンを構成する頂点のインデックスリスト

    Returns:
      list[int]: 反時計回りにしたポリゴン
    """
    polygon = Polygon([
        (self._image_positions[point_id].x, self._image_positions[point_id].y)
        for point_id in inner_polygon
    ])
    if not polygon.exterior.is_ccw:
      # 時計回りなら反転させる
      return inner_polygon[::-1]

    return inner_polygon

  def _calculate_slope(self, point1, point2) -> float:
    """
    2点間の傾きを計算する
    Args:
      point1, point2: 2点の座標 [x, y]

    Returns:
      float: i軸に対する傾き（角度）
    """
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    return np.arctan2(dy, dx) * 180 / np.pi  # ラジアンから度に変換

  def _calculate_angle_between_points(self, vertices: list[Point], point_id: int, polygon: list[int]):
    """
    ポリゴンの指定した頂点を基準に前後の頂点との角度を計算する
    Args:
      vertices (list[Point]): 全頂点の座標リスト
      point_id (int): 基準となる頂点のインデックス
      polygon (list[int]): ポリゴンを構成する頂点のインデックスリスト

    Returns:
      float: 頂点の角度（度数法）
    """
    prev_point_id = -1
    next_point_id = -1
    for list_index, point_id_in_polygon in enumerate(polygon):
      if point_id_in_polygon == point_id:
        prev_list_index = len(polygon) - 1 if (list_index - 1) == -1 else list_index - 1
        next_list_index = 0 if (list_index + 1) == len(polygon) else list_index + 1

        prev_point_id = polygon[prev_list_index]
        next_point_id = polygon[next_list_index]

    # 現在の頂点
    current_point = vertices[point_id]
    # 前の頂点
    prev_point = vertices[prev_point_id]
    # 次の頂点
    next_point = vertices[next_point_id]

    # 前の頂点と次の頂点の傾きをそれぞれ計算
    prev_slope = self._calculate_slope(current_point, prev_point)
    next_slope = self._calculate_slope(current_point, next_point)

    # 前の頂点の角度 - 次の頂点の角度
    angle = prev_slope - next_slope

    # 角度が負の場合は360度を足す
    if angle < 0:
      angle += 360

    return angle

  def _find_vertices_with_angle_over_200(self, inner_polygon: list[int]):
    """
    ポリゴンの内部から見て200度以上の角度を持つ頂点を探す
    Args:
      inner_polygon (list[int]): ポリゴンを構成する頂点のインデックスリスト

    Returns:
      list[int]: 200度以上の角度を持つ頂点のインデックスのリスト
    """
    counter_clockwised_polygon = self._ensure_counterclockwise(inner_polygon)
    vertices_with_large_angles: list[int] = []

    for point_id in counter_clockwised_polygon:
      angle = self._calculate_angle_between_points(
          self._image_positions,
          point_id,
          counter_clockwised_polygon,
      )

      if angle > 200:
        vertices_with_large_angles.append(point_id)

    return vertices_with_large_angles

  def _bresenham_line(self, point_a: list[int], point_b: list[int]):
    """
    Bresenhamのアルゴリズムを使って2点間の直線を描画する。

    Args:
        point_a (list[int]): 始点のピクセル座標
        i1, j1 (int): 終点のピクセル座標

    Returns:
        list[list[int]]: 線上のすべてのピクセル座標
    """
    i0, j0 = point_a
    i1, j1 = point_b
    pixels = []
    dx = abs(i1 - i0)
    dy = abs(j1 - j0)
    sx = 1 if i0 < i1 else -1
    sy = 1 if j0 < j1 else -1
    err = dx - dy

    while True:
      pixels.append([i0, j0])
      if i0 == i1 and j0 == j1:
        break
      e2 = 2 * err
      if e2 > -dy:
        err -= dy
        i0 += sx
      if e2 < dx:
        err += dx
        j0 += sy

    return pixels

  def _get_polygon_line_pixel_positions(self, inner_polygon: list[int]):
    """
    ポリゴンのすべての辺のピクセル座標を取得する。

    Args:
        inner_polygon (list[int]): ポリゴンを構成する頂点のインデックスリスト

    Returns:
        list[tuple[int, int]]: ポリゴンを構成するすべての辺上のピクセル座標
    """

    inner_polygon_ij = self._get_polygon_ij(inner_polygon)
    polygon_line_pixel_positions = []
    num_points = len(inner_polygon_ij)

    for index in range(num_points):
      # 頂点iと頂点i+1を結ぶ線を引く（最後の頂点は最初の頂点と結ぶ）
      polygon_line_pixel_positions.extend(self._bresenham_line(
          inner_polygon_ij[index],
          inner_polygon_ij[(index + 1) % num_points]
      ))

    return polygon_line_pixel_positions

  def _is_devidable_poligon(self, inner_polygon: list[int]):
    if len(inner_polygon) <= 4:
      return False

    layer_number_points_pair = self._get_layer_number_points_pair(inner_polygon)
    majority_layer_number = self._get_majority_layer_number(layer_number_points_pair)
    if majority_layer_number == RoofLayerInfo.NOISE_POINT:
      return False

    noise_ijs = layer_number_points_pair.get(RoofLayerInfo.NOISE_POINT) or []
    noise_count = len(noise_ijs)
    total_count = sum(len(v) for v in layer_number_points_pair.values()) - noise_count
    if total_count < 30:
      return False

    majority_layer_count = len(layer_number_points_pair[majority_layer_number])
    majority_layer_rate = majority_layer_count / total_count
    if total_count == 0 or majority_layer_rate > 0.85:
      return False

    return True

  def _get_devidable_point_on_poligon(self, inner_polygon: list[int]):
    """
    ポリゴン分割できる開始線の開始頂点IDと終了点の座標(i,j)リストのペアを出す。

    Args:
        inner_polygon (list[int]): ポリゴンを構成する頂点のインデックスリスト

    Returns:
        dict[int, list[list[int]]]: ポリゴン分割できる開始線の開始頂点IDと終了点の座標(i,j)リストのペア
    """

    start_point_id_end_point_ijs_pair: dict[int, list[list[int]]] = {}

    start_point_ids = self._find_vertices_with_angle_over_200(inner_polygon)
    polygon_line_pixel_ijs = self._get_polygon_line_pixel_positions(inner_polygon)

    inner_polygon_ij = self._get_polygon_ij(inner_polygon)
    roof_polygon_devidable = np.full((self._height, self._width), RoofLayerInfo.NO_POINT, dtype=np.int_)
    poly = Polygon(inner_polygon_ij)

    for index, current_start_point_id in enumerate(start_point_ids):
      start_point_ids_length = len(start_point_ids)
      prev_start_point_id = start_point_ids[index - 1]
      next_start_point_id = start_point_ids[(index + 1) % start_point_ids_length]

      prev_start_point_ij, current_start_point_ij, next_start_point_ij = self._get_polygon_ij([
          prev_start_point_id, current_start_point_id, next_start_point_id
      ])

      available_end_poins_ijs: list[list[int]] = []
      for end_point_ij in polygon_line_pixel_ijs:
        prev_polygon_line = self._bresenham_line(current_start_point_ij, prev_start_point_ij)
        next_polygon_line = self._bresenham_line(current_start_point_ij, next_start_point_ij)
        if end_point_ij in prev_polygon_line:
          continue

        if end_point_ij in next_polygon_line:
          continue

        line_points = self._bresenham_line(current_start_point_ij, end_point_ij)
        line_points_without_poligon_outer_points = [
            line_point for line_point in line_points
            if line_point not in polygon_line_pixel_ijs
        ]

        can_select_point = True
        for line_point in line_points_without_poligon_outer_points:
          if poly.contains(GeoPoint(line_point[0], line_point[1])) is False:
            can_select_point = False

        if len(line_points_without_poligon_outer_points) == 0:
          can_select_point = False

        if can_select_point:
          available_end_poins_ijs.append(end_point_ij)

      start_point_id_end_point_ijs_pair[current_start_point_id] = available_end_poins_ijs

      if self._debug_mode:
        for i, j in polygon_line_pixel_ijs:
          roof_polygon_devidable[i, j] = RoofLayerInfo.NOISE_POINT

        for i, j in available_end_poins_ijs:
          roof_polygon_devidable[i, j] = RoofLayerInfo.ROOF_LINE_POINT

        roof_polygon_devidable[current_start_point_ij[0], current_start_point_ij[1]] = RoofLayerInfo.ROOF_LINE_POINT
        self._roof_layer_info.save_layer_image(
            roof_polygon_devidable,
            f"roof_polygon_devidable_from_{current_start_point_ij[0]}_{current_start_point_ij[1]}.png",
        )

    return start_point_id_end_point_ijs_pair
