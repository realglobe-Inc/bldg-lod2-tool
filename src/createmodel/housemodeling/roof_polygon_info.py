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
      self.polygon_edges_for_debug_image = set()
      self.has_too_much_noise_on_dsm = False
      self._process_polygons()

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
      inner_polygon_ij = self._get_polygon_ij(inner_polygon)
      poly = Polygon(inner_polygon_ij)
      self._update_polygon_edges_for_debug_image(poly)
      self._process_polygon_layers(poly)

    self._save_debug_images()

  def _update_polygon_edges_for_debug_image(self, poly: Polygon):
    """
    ポリゴンの外周線を取得し、デバッグ用のエッジリストに追加する。

    Args:
        poly (Polygon): ShapelyのPolygonオブジェクト
    """
    coords = list(poly.exterior.coords)  # ポリゴンの外周座標リスト
    for i in range(len(coords) - 1):  # -1 は最後のエッジを無視するため
      sorted_coord = tuple(sorted([coords[i], coords[i + 1]]))  # 重複を防ぐため、sort
      self.polygon_edges_for_debug_image.add(sorted_coord)

  def _process_polygon_layers(self, poly: Polygon):
    """
    ポリゴン内のレイヤー情報を処理し、ノイズを判定する。

    Args:
        poly (Polygon): ShapelyのPolygonオブジェクト
    """
    layer_number_points_pair = self._get_layer_number_points_pair(poly)
    layer_number = self._get_majority_layer_number(layer_number_points_pair)

    if layer_number < 0:
      self.has_too_much_noise_on_dsm = True

    self._update_rgb_images(layer_number_points_pair, layer_number)

  def _get_layer_number_points_pair(self, poly: Polygon):
    """
    ポリゴン内のレイヤー番号と対応するポイントを取得する。

    Args:
        poly (Polygon): ShapelyのPolygonオブジェクト

    Returns:
        dict[int, list[tuple[int, int]]]: レイヤー番号とそのポイント(i, j)のペアを含む辞書
    """
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
    self._roof_layer_info.save_roof_line_image(self._roof_layer_info.rgb_image, self.polygon_edges_for_debug_image)
    self._roof_layer_info.save_roof_line_image(
        self._rgb_image_of_roof_line_with_layer_class_as_is,
        self.polygon_edges_for_debug_image,
        'roof_line_with_layer_class_as_is.png',
    )
    self._roof_layer_info.save_roof_line_image(
        self._rgb_image_of_roof_line_with_layer_class_to_be,
        self.polygon_edges_for_debug_image,
        'roof_line_with_layer_class_to_be.png',
    )
    self._roof_layer_info.save_roof_line_image(
        self._rgb_image_of_roof_line_with_layer_class,
        self.polygon_edges_for_debug_image,
        'roof_line_with_layer_class.png',
    )
