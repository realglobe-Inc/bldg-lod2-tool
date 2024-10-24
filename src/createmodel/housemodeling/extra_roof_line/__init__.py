import numpy as np
from shapely.geometry import Polygon

from ....createmodel.createmodelexception import CreateModelException
from ....createmodel.message import CreateModelMessage
from .polygon_devision import PolygonDevision
from ..model_surface_creation.utils.geometry import Point
from ..roof_layer_info import RoofLayerInfo


class ExtraRoofLine:
  @property
  def inner_polygon_ijs_list_after(self):
    """
    Returns:
      list[list[tuple[float, float]]]: 分離されたポリゴンリスト
    """

    return self._inner_polygon_ijs_list_after

  @property
  def has_splited_polygon(self):
    """
    Returns:
      bool: 分離されたポリゴンリストある場合 True
    """

    return self._has_splited_polygon

  def __init__(
      self,
      cartesian_points: list[Point],
      inner_polygons: list[list[int]],
      roof_layer_info: RoofLayerInfo,
      debug_mode: bool = False
  ):
    """
    HEAT が出せてない屋根線を追加

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
    self._dsm_grid_rgbs_origin = self._roof_layer_info.dsm_grid_rgbs

    # Cartesian座標と対応する画像座標の対応付け
    self._image_vertices: list[Point] = self._get_image_vertices(self._roof_layer_info, self._cartesian_vertices)

    inner_polygon_ijs_list_before: list[list[tuple[int, int]]] = []
    for inner_polygon in self._inner_polygons:
      inner_polygon_ijs_list = [
          PolygonDevision.point_id_to_ij(self._image_vertices, point_id) for point_id in inner_polygon
      ]
      inner_polygon_ijs_list_before.append(inner_polygon_ijs_list)

    if debug_mode:
      self._save_roof_line_with_layer_class_images(
          inner_polygon_ijs_list_before,
          'roof_line_with_layer_class_step_1_origin_polygons.png',
          'roof_line_with_layer_class_step_2_origin_roof_layers.png',
          'roof_line_with_layer_class_step_3_filled_origin_polygons.png',
      )

    self._inner_polygon_ijs_list_after: list[list[tuple[float, float]]] = []
    self._has_splited_polygon = False

    for inner_polygon in self._inner_polygons:
      polygon_devision = PolygonDevision(
          vertices=self._image_vertices,
          polygon=inner_polygon,
          roof_layer_info=self._roof_layer_info,
          debug_mode=self._debug_mode,
      )
      if (polygon_devision.can_split()):
        splited_polygon_ijs = polygon_devision.get_splited_polygon_ijs(
            'roof_line_with_layer_class_step_4_splited_polygon.png',
        )
        self._inner_polygon_ijs_list_after.extend(splited_polygon_ijs)
        self._has_splited_polygon = True
      else:
        inner_polygon_ijs = [
            PolygonDevision.point_id_to_ij(self._image_vertices, point_id) for point_id in inner_polygon
        ]
        self._inner_polygon_ijs_list_after.append(inner_polygon_ijs)

    if debug_mode:
      self._save_roof_line_with_layer_class_images(
          self._inner_polygon_ijs_list_after,
          'roof_line_with_layer_class_step_5_splited_polygons.png',
          'roof_line_with_layer_class_step_6_splited_roof_layers.png',
          'roof_line_with_layer_class_step_7_filled_splited_polygons.png',
      )

    for polygon_ijs in self._inner_polygon_ijs_list_after:
      if not Polygon(polygon_ijs).is_valid:
        raise CreateModelException(CreateModelMessage.ERR_POLYGON_DIVISION_FAIL)

  def _has_too_many_noise(self):
    has_too_many_noise = False
    for polygon_ijs in self._inner_polygon_ijs_list_after:
      layer_number_point_ijs_pair = PolygonDevision.get_layer_number_grid_ijs_pair(self._roof_layer_info, polygon_ijs)
      noise_point_ijs = layer_number_point_ijs_pair.get(RoofLayerInfo.NOISE_POINT) or []
      point_ijs_count_noise = len(noise_point_ijs)
      point_ijs_count_all = 0
      for point_ijs in layer_number_point_ijs_pair.values():
        point_ijs_count_all += len(point_ijs)

      if point_ijs_count_all >= 25 and (point_ijs_count_noise / point_ijs_count_all) > 0.4:
        print(point_ijs_count_all, point_ijs_count_noise)
        has_too_many_noise = True

    return has_too_many_noise

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

  def _save_roof_line_with_layer_class_images(
      self,
      inner_polygon_ijs_list: list[list[tuple[int, int]]],
      file_name_1: str,
      file_name_2: str,
      file_name_3: str,
  ):
    """
    屋根の壁処理のためのデバッグ用の画像を保存する。

    Args:
      inner_polygon_ijs_list (list[list[tuple[int, int]]]): ポリゴンリスト
      file_name_1 (str): デバッグイメージのファイル名
      file_name_2 (str): デバッグイメージのファイル名
      file_name_3 (str): デバッグイメージのファイル名
    """

    # RGBイメージの初期化
    dsm_grid_rgbs_of_roof_line_with_layer_class_as_is = np.full((self._height, self._width, 3), 255, dtype=np.uint8)
    dsm_grid_rgbs_of_roof_line_with_layer_class_to_be = np.full((self._height, self._width, 3), 255, dtype=np.uint8)
    polygon_edges_for_debug_image = set()

    for polygon_ijs in inner_polygon_ijs_list:
      poly = Polygon(polygon_ijs)
      coords = list(poly.exterior.coords)  # ポリゴンの外周座標リスト
      for i in range(len(coords) - 1):  # -1 は最後のエッジを無視するため
        sorted_coord = tuple(sorted([coords[i], coords[i + 1]]))  # 重複を防ぐため、sort
        polygon_edges_for_debug_image.add(sorted_coord)

    for polygon_ijs in inner_polygon_ijs_list:
      layer_number_point_ijs_pair = PolygonDevision.get_layer_number_grid_ijs_pair(self._roof_layer_info, polygon_ijs)
      majority_layer_number = PolygonDevision.get_majority_layer_number(layer_number_point_ijs_pair)
      for layer_number, layer_points_ij in layer_number_point_ijs_pair.items():
        for i, j in layer_points_ij:
          dsm_grid_rgbs_of_roof_line_with_layer_class_as_is[i, j] = self._roof_layer_info.get_color(layer_number)
          dsm_grid_rgbs_of_roof_line_with_layer_class_to_be[i, j] = self._roof_layer_info.get_color(majority_layer_number)

    self._roof_layer_info.save_roof_line_image(
        self._roof_layer_info.dsm_grid_rgbs,
        polygon_edges_for_debug_image,
        file_name_1,
    )
    self._roof_layer_info.save_roof_line_image(
        dsm_grid_rgbs_of_roof_line_with_layer_class_as_is,
        polygon_edges_for_debug_image,
        file_name_2,
    )
    self._roof_layer_info.save_roof_line_image(
        dsm_grid_rgbs_of_roof_line_with_layer_class_to_be,
        polygon_edges_for_debug_image,
        file_name_3,
    )
