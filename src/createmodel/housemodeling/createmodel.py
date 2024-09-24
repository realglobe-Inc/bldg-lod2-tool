import os
from typing import Optional

from shapely.geometry import Polygon
import numpy as np


from .roof_polygon_info import RoofPolygonInfo
from .house_model import HouseModel
from .coordinates_converter import CoordConverterForCartesianAndImagePos
from .model_surface_creation.extract_roof_surface import extract_roof_surface
from .model_surface_creation.optimize_roof_edge import optimize_roof_edge
from .model_surface_creation.utils.geometry import Point
from .balcony_detection import BalconyDetection
from .roof_edge_detection import RoofEdgeDetection
from ..lasmanager import PointCloud
from .preprocess import Preprocess


def CreateModel(
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

  image_size = 256

  # 作成に使用するためのデータを作成
  preprocess = Preprocess(grid_size=grid_size, image_size=image_size, expand_rate=expand_rate, building_id=building_id)
  rgb_image, depth_image, roof_layer_info = preprocess.preprocess(cloud, min_ground_height, shape, debug_mode)

  # 平面直角座標と画像座標の変換を行うクラスを作成
  min_x, min_y = cloud.get_points()[:, :2].min(axis=0)
  max_x, max_y = cloud.get_points()[:, :2].max(axis=0)
  expanded_grid_size = grid_size / (expand_rate if expand_rate is not None else 1)
  height = round((max_y - min_y) / expanded_grid_size) + 1
  width = round((max_x - min_x) / expanded_grid_size) + 1
  cartesian_coord_upper_left = (
      min_x - (image_size - width) / 2 * expanded_grid_size,
      max_y + (image_size - height) / 2 * expanded_grid_size,
  )
  coord_converter = CoordConverterForCartesianAndImagePos(
      grid_size=expanded_grid_size,
      cartesian_coord_upper_left=cartesian_coord_upper_left,
  )

  # 屋根線検出
  roof_edge_detection = RoofEdgeDetection(roof_edge_detection_checkpoint_path, use_gpu)
  image_corners_xy, edges = roof_edge_detection.infer(rgb_image)
  result_edges: list[tuple[int, int]] = []

  # 画像座標から平面直角座標への変換
  cartesian_corners_xy = np.array([
      coord_converter.image_point_to_cartesian_point(x, y) for x, y in image_corners_xy
  ])

  # LoD2モデルデータの作成
  cartesian_points, inner_edge, outer_edge = optimize_roof_edge(shape, cartesian_corners_xy, edges)
  result_edges = inner_edge + outer_edge
  outer_polygon, inner_polygons = extract_roof_surface(cartesian_points, result_edges)
  roof_polygon_info = RoofPolygonInfo(cartesian_points, inner_polygons, roof_layer_info, debug_mode)

  # 平面直角座標から画像座標への変換
  image_points = [
      Point(*coord_converter.cartesian_point_to_image_point(cartesian_point.x, cartesian_point.y))
      for cartesian_point in cartesian_points
  ]

  # バルコニーセグメンテーション
  balcony_detection = BalconyDetection(balcony_segmentation_checkpoint_path, use_gpu)
  balcony_flags = balcony_detection.infer(
      rgb_image=rgb_image,
      depth_image=depth_image,
      image_points=image_points,
      polygons=inner_polygons,
      threshold=0.5
  )
  cartesian_points_xy = np.array([(point.x, point.y) for point in cartesian_points])

  # 3Dモデルの生成
  model = HouseModel(id=building_id, shape=shape)
  model.create_model_surface(
      point_cloud=cloud.get_points().copy(),
      points_xy=cartesian_points_xy,
      inner_polygons=inner_polygons,
      outer_polygon=outer_polygon,
      ground_height=min_ground_height,
      balcony_flags=balcony_flags
  )
  model.simplify(threshold=5)

  # 壁面非水密エラー修正
  model.rectify()

  # objファイルの作成
  file_name = f'{building_id}.obj'
  obj_path = os.path.join(output_folder_path, file_name)
  model.output_obj(path=obj_path)
