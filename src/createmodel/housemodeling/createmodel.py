import os
from typing import Optional

from shapely.geometry import Polygon
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .extra_roof_line import ExtraRoofLine
from .house_model import HouseModel
from .coordinates_converter import CoordConverterForCartesianAndImagePos
from .model_surface_creation.extract_roof_surface import extract_roof_surface
from .model_surface_creation.optimize_roof_edge import optimize_roof_edge
from .model_surface_creation.utils.geometry import Point
from .balcony_detection import BalconyDetection
from .roof_edge_detection import RoofEdgeDetection
from ..lasmanager import PointCloud
from .preprocess import Preprocess


class CreateHouseModel:
  """複雑形状家屋の3Dモデルを作成"""

  def __init__(
      self,
      cloud: PointCloud,
      shape: Polygon,
      building_id: str,
      min_ground_height: float,
      output_folder_path: str,
      balcony_segmentation_checkpoint_path: str,
      roof_edge_detection_checkpoint_path: str,
      grid_size: float = 0.25,
      expand_rate: Optional[float] = None,
      use_gpu: bool = False,
      debug_mode: bool = False,
  ) -> None:
    """家屋3Dモデルの作成

    Args:
      cloud(PointCloud): 建物点群
      shape(Polygon): 建物外形ポリゴン
      bulding_id(str): 建物ID
      min_ground_height(float): 最低地面の高さ
      output_folder_path(str): 出力先フォルダ
      balcony_segmentation_checkpoint_path(str): バルコニーのセグメンテーションの学習済みモデルファイルパス
      roof_edge_detection_checkpoint_path(str): 屋根線検出の学習済みモデルファイルパス
      grid_size(float,optional): 点群の間隔(meter) (Default: 0.25),
      expand_rate(float, optional): 画像の拡大率 (Default: 1),
      use_gpu(bool, optional): 推論時のGPU使用の有無 (Default: False)
      debug_mode (bool, optional): デバッグモード (Default: False)
    """

    self._image_size = 256

    self._cloud = cloud
    self._shape = shape
    self._building_id = building_id
    self._min_ground_height = min_ground_height
    self._output_folder_path = output_folder_path
    self._balcony_segmentation_checkpoint_path = balcony_segmentation_checkpoint_path
    self._roof_edge_detection_checkpoint_path = roof_edge_detection_checkpoint_path
    self._grid_size = grid_size
    self._expand_rate = expand_rate
    self._use_gpu = use_gpu
    self._debug_mode = debug_mode

    # 作成に使用するためのデータを作成
    preprocess = Preprocess(grid_size=grid_size, image_size=self._image_size, expand_rate=expand_rate, building_id=building_id)
    result_preprocess = preprocess.preprocess(self._cloud, min_ground_height, shape, debug_mode)
    self._square_dsm_grid_rgbs, self._depth_image, self._roof_layer_info = result_preprocess

    self._coord_converter = self._get_coord_converter()
    roof_edge_detection = RoofEdgeDetection(self._roof_edge_detection_checkpoint_path, self._use_gpu)
    roof_vertice_ijs_tmp, roof_edges_tmp = roof_edge_detection.infer(self._square_dsm_grid_rgbs)
    roof_cartesian_points_tmp, outer_polygon_tmp, inner_polygons_tmp = self._get_roof_polygons(
        roof_vertice_ijs_tmp, roof_edges_tmp
    )

    extra_roof_line = ExtraRoofLine(
        roof_cartesian_points_tmp,
        inner_polygons_tmp,
        self._roof_layer_info,
        debug_mode,
    )
    if extra_roof_line.has_splited_polygon:
      roof_vertice_ijs = list(set([
          polygon_ij
          for polygon_ijs in extra_roof_line.inner_polygon_ijs_list_after
          for polygon_ij in polygon_ijs
      ]))
      vertex_point_id_pair = {roof_vertice_ij: index for index, roof_vertice_ij in enumerate(roof_vertice_ijs)}

      polygon_edges: set[tuple[int, int]] = set()
      for polygon_ijs in extra_roof_line.inner_polygon_ijs_list_after:
        poly = Polygon(polygon_ijs)
        coords = list(poly.exterior.coords)  # ポリゴンの外周座標リスト
        for i in range(len(coords) - 1):  # -1 は最後のエッジを無視するため
          point_id_a = vertex_point_id_pair[coords[i]]
          point_id_b = vertex_point_id_pair[coords[i + 1]]
          sorted_edge: tuple[int, int] = tuple(sorted([point_id_a, point_id_b]))  # 重複を防ぐため、sort
          polygon_edges.add(sorted_edge)

      roof_edges = list(polygon_edges)

      result_edges: list[tuple[int, int]] = []
      tmp_roof_vertice_xys = np.array([self._ij_to_xy(i, j) for i, j in roof_vertice_ijs])

      # LoD2モデルデータの作成
      roof_cartesian_points, inner_edge, outer_edge = optimize_roof_edge(
          self._shape,
          tmp_roof_vertice_xys,
          roof_edges,
      )
      result_edges = inner_edge + outer_edge
      outer_polygon, inner_polygons = extract_roof_surface(roof_cartesian_points, result_edges)
    else:
      roof_cartesian_points = roof_cartesian_points_tmp
      outer_polygon = outer_polygon_tmp
      inner_polygons = inner_polygons_tmp

    self._balcony_flags = self._get_balcony_flags(roof_cartesian_points, inner_polygons)

    self._create_model(
        roof_points=roof_cartesian_points,
        inner_polygons=inner_polygons,
        outer_polygon=outer_polygon,
        balcony_flags=self._balcony_flags,
    )

  def _get_coord_converter(self):
    """
    座標変換 : 画像 image pos(i,j) <-> DSM(x,y)
    """
    min_x, min_y = self._cloud.get_points()[:, :2].min(axis=0)
    max_x, max_y = self._cloud.get_points()[:, :2].max(axis=0)
    expanded_grid_size = self._grid_size / (self._expand_rate if self._expand_rate is not None else 1)
    height = round((max_y - min_y) / expanded_grid_size) + 1
    width = round((max_x - min_x) / expanded_grid_size) + 1
    cartesian_coord_upper_left = (
        min_x - (self._image_size - width) / 2 * expanded_grid_size,
        max_y + (self._image_size - height) / 2 * expanded_grid_size,
    )
    coord_converter = CoordConverterForCartesianAndImagePos(
        grid_size=expanded_grid_size,
        cartesian_coord_upper_left=cartesian_coord_upper_left,
    )

    return coord_converter

  def _get_roof_polygons(
      self,
      roof_vertice_ijs: list[tuple[float, float]],
      roof_edges: list[tuple[int, int]],
  ):
    result_edges: list[tuple[int, int]] = []

    # 画像座標から平面直角座標への変換
    tmp_roof_vertice_xys = np.array([
        self._coord_converter.image_point_to_cartesian_point(i, j) for i, j in roof_vertice_ijs
    ])

    # LoD2モデルデータの作成
    optimized_roof_points, inner_edge, outer_edge = optimize_roof_edge(self._shape, tmp_roof_vertice_xys, roof_edges)
    result_edges = inner_edge + outer_edge
    outer_polygon, inner_polygons = extract_roof_surface(optimized_roof_points, result_edges)

    return optimized_roof_points, outer_polygon, inner_polygons

  def _get_balcony_flags(
      self,
      roof_points: list[Point],
      inner_polygons: list[int],
  ):
    # 平面直角座標から画像座標への変換
    image_points = [
        Point(*self._coord_converter.cartesian_point_to_image_point(cartesian_point.x, cartesian_point.y))
        for cartesian_point in roof_points
    ]

    # バルコニーセグメンテーション
    balcony_detection = BalconyDetection(self._balcony_segmentation_checkpoint_path, self._use_gpu)
    balcony_flags = balcony_detection.infer(
        dsm_grid_rgbs=self._square_dsm_grid_rgbs,
        depth_image=self._depth_image,
        image_points=image_points,
        polygons=inner_polygons,
        threshold=0.5
    )

    return balcony_flags

  def _create_model(
      self,
      roof_points: list[Point],
      inner_polygons: list[int],
      outer_polygon: list[int],
      balcony_flags: list[bool],
  ):
      # 3Dモデルの生成
    model = HouseModel(id=self._building_id, shape=self._shape)
    model.create_model_surface(
        point_cloud=self._cloud.get_points().copy(),
        points_xy=np.array([(point.x, point.y) for point in roof_points]),
        inner_polygons=inner_polygons,
        outer_polygon=outer_polygon,
        ground_height=self._min_ground_height,
        balcony_flags=balcony_flags
    )
    model.simplify(threshold=5)

    # 壁面非水密エラー修正
    model.rectify()

    # objファイルの作成
    file_name = f'{self._building_id}.obj'
    obj_path = os.path.join(self._output_folder_path, file_name)
    model.output_obj(path=obj_path)

  def _ij_to_xy(self, i: float, j: float):
    """
    画像座標 (i, j) からDSM点群の (x, y) へ変換

    Args:
      x (float): 選択した任意の点の x
      y (float): 選択した任意の点の y
    """
    # i と j の範囲を定義
    i_values = np.arange(self._roof_layer_info.origin_dsm_grid_xyzs.shape[0])  # i の範囲
    j_values = np.arange(self._roof_layer_info.origin_dsm_grid_xyzs.shape[1])  # j の範囲

    # 各次元の座標 (x, y) に対して補間関数を作成
    x_coords = self._roof_layer_info.origin_dsm_grid_xyzs[:, :, 0]
    y_coords = self._roof_layer_info.origin_dsm_grid_xyzs[:, :, 1]

    # x, y それぞれに対して補間関数を作成
    interp_x = RegularGridInterpolator((i_values, j_values), x_coords)
    interp_y = RegularGridInterpolator((i_values, j_values), y_coords)
    x = float(interp_x((i, j)))
    y = float(interp_y((i, j)))

    return x, y
