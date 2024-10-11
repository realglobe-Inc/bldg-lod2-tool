import numpy as np
from shapely.geometry import Polygon

from .polygon_devision import (
    bresenham_line,
    get_layer_number_point_ijs_pair,
    get_majority_layer_number,
    get_new_intersection_polygon_ijs,
    is_splitable_poligon,
    point_id_to_ij,
)
from ..model_surface_creation.utils.geometry import Point
from ..roof_layer_info import RoofLayerInfo


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
    self._cartesian_vertices = cartesian_points
    self._inner_polygons = inner_polygons
    self._roof_layer_info = roof_layer_info
    self._debug_mode = debug_mode
    self._height, self._width = self._roof_layer_info.layer_class.shape
    self._layer_class_origin = self._roof_layer_info.layer_class
    self._rgb_image_origin = self._roof_layer_info.rgb_image

    # Cartesian座標と対応する画像座標の対応付け
    self._image_vertices: list[Point] = self._get_image_vertices(self._roof_layer_info, self._cartesian_vertices)

    # RGBイメージの初期化
    self._rgb_image_of_roof_line_with_layer_class_as_is = np.full((self._height, self._width, 3), 255, dtype=np.uint8)
    self._rgb_image_of_roof_line_with_layer_class_to_be = np.full((self._height, self._width, 3), 255, dtype=np.uint8)
    self._rgb_image_of_roof_line_with_layer_class = np.full((self._height, self._width, 3), 255, dtype=np.uint8)

    self._polygon_edges_for_debug_image = set()
    self.has_too_much_noise_on_dsm = False
    self._process_polygons(self._image_vertices, self._inner_polygons)

    new_inner_polygons: list[list[int]] = []
    for inner_polygon in self._inner_polygons:
      if is_splitable_poligon(self._image_vertices, inner_polygon, self._layer_class_origin):
        pass
        get_new_intersection_polygon_ijs(
            self._image_vertices,
            inner_polygon,
            self._roof_layer_info,
            self._debug_mode,
        )

      else:
        new_inner_polygons.append(inner_polygon)

  def _get_image_vertices(self, roof_layer_info: RoofLayerInfo, cartesian_points: list[Point]):
    """
    Cartesian座標 (x, y) を画像座標 (i, j) に変換し、2次元リストに保存する。

    Returns:
      list[list[int]]: Cartesian座標に対応する画像座標(i, j)のリスト
    """
    image_vertices: list[Point] = []
    for point in cartesian_points:
      nearest_x, nearest_y = roof_layer_info.find_nearest_xy(point.x, point.y)
      nearest_i, nearest_j = roof_layer_info.xy_to_ij(nearest_x, nearest_y)
      image_vertices.append(Point(nearest_i, nearest_j))

    return image_vertices

  def _process_polygons(self, image_vertices: list[Point], inner_polygons: list[list[int]]):
    """
    内部ポリゴンの処理を行い、各種画像を生成する。
    """
    for inner_polygon in inner_polygons:
      self._update_polygon_edges_for_debug_image(image_vertices, inner_polygon)
      self._process_polygon_layers(image_vertices, inner_polygon)

    if self._debug_mode:
      self._save_debug_images()

  def _update_polygon_edges_for_debug_image(self, image_vertices: list[Point], inner_polygon: list[int]):
    """
    ポリゴンの外周線を取得し、デバッグ用のエッジリストに追加する。

    Args:
      inner_polygon (list[int]): ポリゴンを構成する頂点のインデックスリスト
    """
    inner_polygon_ijs = [point_id_to_ij(image_vertices, point_id) for point_id in inner_polygon]
    poly = Polygon(inner_polygon_ijs)
    coords = list(poly.exterior.coords)  # ポリゴンの外周座標リスト
    for i in range(len(coords) - 1):  # -1 は最後のエッジを無視するため
      sorted_coord = tuple(sorted([coords[i], coords[i + 1]]))  # 重複を防ぐため、sort
      self._polygon_edges_for_debug_image.add(sorted_coord)

  def _process_polygon_layers(self, image_vertices: list[Point], inner_polygon: list[int]):
    """
    ポリゴン内のレイヤー情報を処理し、ノイズを判定する。

    Args:
      inner_polygon (list[int]): ポリゴンを構成する頂点のインデックスリスト
    """
    layer_number_point_ijs_pair = get_layer_number_point_ijs_pair(image_vertices, inner_polygon, self._layer_class_origin)
    majority_layer_number = get_majority_layer_number(layer_number_point_ijs_pair)

    if majority_layer_number < 0:
      self.has_too_much_noise_on_dsm = True

    self._update_rgb_images(layer_number_point_ijs_pair, majority_layer_number)

  def _update_rgb_images(
      self,
      layer_number_point_ijs_pair: dict[int, list[tuple[int, int]]],
      majority_layer_number: int,
  ):
    """
    RGBイメージを更新する。

    Args:
      layer_number_point_ijs_pair (dict[int, list[tuple[int, int]]]): レイヤー番号とそのポイント(i, j)のペアを含む辞書
      majority_layer_number (int): ポリゴン内の多数派のレイヤー番号
    """
    for layer_number, layer_points_ij in layer_number_point_ijs_pair.items():
      for i, j in layer_points_ij:
        self._rgb_image_of_roof_line_with_layer_class_as_is[i, j] = self._roof_layer_info.get_color(layer_number)
        self._rgb_image_of_roof_line_with_layer_class_to_be[i, j] = self._roof_layer_info.get_color(majority_layer_number)
        self._rgb_image_of_roof_line_with_layer_class[i, j] = self._roof_layer_info.get_color(
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

  def _save_roof_polygon_splited_debug_imege(
      self,
      vertices: list[Point],
      start_point_id: int,
      end_point_id: int,
      available_end_point_ijs: list[tuple[int, int]],
      point_ij_edge_point_id_start_end_pair: dict[tuple[int, int], tuple[int, int]],
  ):
    """
    ポリゴン分割の中間課程をイメージとして記録する

    Args:
      vertices (list[Point]): 頂点リスト
      start_point_id (int): ポリゴン分割できる開始線の開始頂点ID
      available_end_point_ijs (list[tuple[int, int]]): ポリゴン分割できる開始線の終了点の座標(i,j)リスト
      point_ij_edge_point_id_start_end_pair (dict[tuple[int, int], tuple[int, int]]): ポリゴンのすべての辺のピクセル座標
    """
    roof_polygon_splitable = np.full((self._height, self._width), RoofLayerInfo.NO_POINT, dtype=np.int_)
    for line_point_ij, _ in point_ij_edge_point_id_start_end_pair.items():
      i, j = line_point_ij
      roof_polygon_splitable[i, j] = RoofLayerInfo.NOISE_POINT

    for i, j in available_end_point_ijs:
      roof_polygon_splitable[i, j] = RoofLayerInfo.ROOF_LINE_POINT

    spliting_list_start_point_ij = point_id_to_ij(vertices, start_point_id)
    spliting_list_end_point_ij = point_id_to_ij(vertices, end_point_id)
    spliting_line_point_ijs = bresenham_line(spliting_list_start_point_ij, spliting_list_end_point_ij)

    for spliting_line_point_ij in spliting_line_point_ijs:
      i, j = spliting_line_point_ij
      roof_polygon_splitable[i, j] = RoofLayerInfo.ROOF_VERTICE_POINT

    self._roof_layer_info.save_layer_image(
        roof_polygon_splitable,
        f"roof_polygon_splited_from_{spliting_list_start_point_ij[0]}_{spliting_list_start_point_ij[1]}.png",
    )
