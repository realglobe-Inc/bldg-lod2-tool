import os
from typing import Optional

from shapely.geometry import Polygon
import numpy as np

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
    tmp_roof_edge_vertice_ijs, tmp_roof_edges = roof_edge_detection.infer(self._square_dsm_grid_rgbs)
    tmp_roof_polygon_vertex_xy_points, tmp_outer_polygon, tmp_inner_polygons = self._get_roof_polygons(
        tmp_roof_edge_vertice_ijs, tmp_roof_edges
    )

    extra_roof_line = ExtraRoofLine(
        tmp_roof_polygon_vertex_xy_points,
        tmp_inner_polygons,
        self._roof_layer_info,
        debug_mode,
    )
    if extra_roof_line.has_splited_polygon:
      vertex_ijs = list(set([
          polygon_ij
          for polygon_ijs in extra_roof_line.inner_polygon_ijs_list_after
          for polygon_ij in polygon_ijs
      ]))
      vertex_ij_point_id_pair = {vertex_ij: index for index, vertex_ij in enumerate(vertex_ijs)}

      polygon_edges: set[tuple[int, int]] = set()
      for polygon_ijs in extra_roof_line.inner_polygon_ijs_list_after:
        poly = Polygon(polygon_ijs)
        coords = list(poly.exterior.coords)  # ポリゴンの外周座標リスト
        for i in range(len(coords) - 1):  # -1 は最後のエッジを無視するため
          point_id_a = vertex_ij_point_id_pair[coords[i]]
          point_id_b = vertex_ij_point_id_pair[coords[i + 1]]
          sorted_edge: tuple[int, int] = tuple(sorted([point_id_a, point_id_b]))  # 重複を防ぐため、sort
          polygon_edges.add(sorted_edge)

      roof_edges = list(polygon_edges)

      result_edges: list[tuple[int, int]] = []
      tmp_roof_vertex_xys = np.array([self._ij_to_xy(i, j) for i, j in vertex_ijs])
      # LoD2モデルデータの作成
      roof_polygon_vertex_xy_points, inner_edge, outer_edge = optimize_roof_edge(
          self._shape,
          tmp_roof_vertex_xys,
          roof_edges,
      )
      result_edges = inner_edge + outer_edge
      outer_polygon, inner_polygons = extract_roof_surface(roof_polygon_vertex_xy_points, result_edges)
      for inner_polygon in inner_polygons:
        polygon_xys = [
            (roof_polygon_vertex_xy_points[point_id].x, roof_polygon_vertex_xy_points[point_id].y)
            for point_id in inner_polygon
        ]
        if not Polygon(polygon_xys).is_valid:
          breakpoint()
    else:
      roof_polygon_vertex_xy_points = tmp_roof_polygon_vertex_xy_points
      outer_polygon = tmp_outer_polygon
      inner_polygons = tmp_inner_polygons

    # roof_polygon_vertex_xy_points = tmp_roof_polygon_vertex_xy_points
    # outer_polygon = tmp_outer_polygon
    # inner_polygons = tmp_inner_polygons

    self._balcony_flags = self._get_balcony_flags(roof_polygon_vertex_xy_points, inner_polygons)

    self._create_model(
        roof_polygon_vertex_xy_points=roof_polygon_vertex_xy_points,
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
    tmp_roof_vertex_xys = np.array([
        self._coord_converter.image_point_to_cartesian_point(i, j) for i, j in roof_vertice_ijs
    ])

    # LoD2モデルデータの作成
    optimized_roof_polygon_vertex_xy_points, inner_edge, outer_edge = optimize_roof_edge(self._shape, tmp_roof_vertex_xys, roof_edges)
    result_edges = inner_edge + outer_edge
    outer_polygon, inner_polygons = extract_roof_surface(optimized_roof_polygon_vertex_xy_points, result_edges)

    return optimized_roof_polygon_vertex_xy_points, outer_polygon, inner_polygons

  def _get_balcony_flags(
      self,
      roof_polygon_vertex_xy_points: list[Point],
      inner_polygons: list[int],
  ):
    # 平面直角座標から画像座標への変換
    image_points = [
        Point(*self._coord_converter.cartesian_point_to_image_point(cartesian_point.x, cartesian_point.y))
        for cartesian_point in roof_polygon_vertex_xy_points
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
      roof_polygon_vertex_xy_points: list[Point],
      inner_polygons: list[int],
      outer_polygon: list[int],
      balcony_flags: list[bool],
  ):
      # 3Dモデルの生成
    model = HouseModel(id=self._building_id, shape=self._shape)
    model.create_model_surface(
        point_cloud=self._cloud.get_points().copy(),
        points_xy=np.array([(point.x, point.y) for point in roof_polygon_vertex_xy_points]),
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
    グリッド間隔の違いを考慮して変換を行う

    Args:
      i (float): 画像座標系の i
      j (float): 画像座標系の j
    """

    # i, j 座標系の端の2点を取得 (例として左上と右下)
    i0, j0 = 0, 0
    i1, j1 = self._roof_layer_info.origin_dsm_grid_xyzs.shape[0] - 1, self._roof_layer_info.origin_dsm_grid_xyzs.shape[1] - 1

    # i0, j0 と i1, j1 に対応する x, y 座標を取得
    x0, y0, _ = self._roof_layer_info.origin_dsm_grid_xyzs[i0, j0]
    x1, y1, _ = self._roof_layer_info.origin_dsm_grid_xyzs[i1, j1]

    # i, j グリッド間隔と x, y グリッド間隔を計算
    grid_spacing_ij = np.array([i1 - i0, j1 - j0])
    grid_spacing_xy = np.array([x1 - x0, y1 - y0])

    # グリッド間隔の比率を計算
    scale_x = grid_spacing_xy[0] / grid_spacing_ij[0]  # x方向のスケール
    scale_y = grid_spacing_xy[1] / grid_spacing_ij[1]  # y方向のスケール

    # i, j 座標系でのベクトル (i0, j0) -> (i1, j1)
    vec_ij = np.array([i1 - i0, j1 - j0])

    # x, y 座標系でのベクトル (x0, y0) -> (x1, y1)
    vec_xy = np.array([x1 - x0, y1 - y0])

    # 内積を使ってベクトル間の角度を計算
    dot_product = np.dot(vec_ij, vec_xy)
    norm_ij = np.linalg.norm(vec_ij)
    norm_xy = np.linalg.norm(vec_xy)

    # (i, j)座標系と(x, y)座標系の角度計算
    cos_theta = dot_product / (norm_ij * norm_xy)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # np.clip で丸める

    # 符号を確認するために外積を使用
    cross_product = np.cross(vec_ij, vec_xy)
    if cross_product < 0:
      theta = -theta

    # i, j から x, y に変換する際、スケーリングを適用して回転
    x_scaled = i * scale_x
    y_scaled = j * scale_y

    # 回転行列を適用して (i, j) を (x', y') に変換
    x_rot = x_scaled * np.cos(theta) - y_scaled * np.sin(theta)
    y_rot = x_scaled * np.sin(theta) + y_scaled * np.cos(theta)

    # 回転後に原点 (x0, y0) のシフトを適用
    x = x_rot + x0
    y = y_rot + y0

    return x, y
