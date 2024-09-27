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

    if self._debug_mode:
      self._polygon_edges_for_debug_image = set()
      self.has_too_much_noise_on_dsm = False
      self._process_polygons()

      for inner_polygon in self._inner_polygons:
        print(self._find_vertices_with_angle_over_200(inner_polygon))
        # self._get_polygon_line_pixel_positions(inner_polygon)

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
    layer_number = self._get_majority_layer_number(layer_number_points_pair)

    if layer_number < 0:
      self.has_too_much_noise_on_dsm = True

    self._update_rgb_images(layer_number_points_pair, layer_number)

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
      layer_number: int,
  ):
    """
    RGBイメージを更新する。

    Args:
      layer_number_points_pair (dict[int, list[tuple[int, int]]]): レイヤー番号とそのポイント(i, j)のペアを含む辞書
      layer_number (int): ポリゴン内の多数派のレイヤー番号
    """
    for layer_number, layer_points_ij in layer_number_points_pair.items():
      for i, j in layer_points_ij:
        self._rgb_image_of_roof_line_with_layer_class_as_is[i, j] = self._roof_layer_info._get_color(layer_number)
        self._rgb_image_of_roof_line_with_layer_class_to_be[i, j] = self._roof_layer_info._get_color(layer_number)
        self._rgb_image_of_roof_line_with_layer_class[i, j] = self._roof_layer_info._get_color(
            RoofLayerInfo.NOISE_POINT if layer_number != layer_number else self._layer_class_origin[i, j]
        )

  def _save_debug_images(self):
    """
    デバッグ用の画像を保存する。
    """
    self._roof_layer_info.save_roof_line_image(self._roof_layer_info.rgb_image, self._polygon_edges_for_debug_image)
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
    vertices_with_large_angles = []

    for point_id in counter_clockwised_polygon:
      angle = self._calculate_angle_between_points(
          self._image_positions,
          point_id,
          counter_clockwised_polygon,
      )

      if angle > 200:
        vertices_with_large_angles.append(point_id)

    return vertices_with_large_angles
