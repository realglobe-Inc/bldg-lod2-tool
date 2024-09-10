from typing import Union

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Point, Polygon
from scipy.spatial import Delaunay

from src.util.objinfo import BldElementType, ObjInfo


def calculate_centroid(triangle: list[int], vertices: NDArray[np.float_]):
  """Calculate the centroid of a triangle given by indices of its vertices."""
  vertex_coords = np.array([vertices[idx] for idx in triangle])
  centroid = np.mean(vertex_coords, axis=0)
  return centroid


def is_point_in_polygon(
    point: Union[list[float], NDArray[np.float_]],
    polygon: Union[list[list[float]], NDArray[np.float_]],
    holes: Union[list[list[list[float]]], list[NDArray[np.float_]], NDArray[np.float_]] = []
):
  """点が多角形内にあるかどうかを判断する簡易的な方法"""
  poly = Polygon(polygon, holes)
  return poly.contains(Point(point))


def calculate_normal(vertices: Union[NDArray[np.float_], list[list[float]]]):
  """頂点から平面の法線を計算"""
  v1, v2, v3 = np.array(vertices[:3], dtype=np.float_)
  return np.cross(v2 - v1, v3 - v1)


def can_be_poligon(vertices: Union[NDArray[np.float_], list[list[float]]]):
  """頂点4個以上のポリゴンが作れるか"""
  for i in range(len(vertices) - 2):
    normal = calculate_normal(vertices[i:i + 3])
    for value in normal:
      if value != 0: return False

  return True


def project_to_best_plane(vertices_3d: list[list[float]]) -> list[list[float]]:
  """法線に基づいて最適な平面に頂点を投影し、距離歪みを補正"""
  normal = calculate_normal(vertices_3d)
  normal_abs = np.abs(normal / np.linalg.norm(normal))

  # 距離歪みを最小化するためのスケーリング係数
  scale_factors = np.sqrt(1 - normal_abs**2)

  np_vertices_3d = np.array(vertices_3d, dtype=np.float_)
  if normal[2] >= normal[1] and normal[2] >= normal[0]:  # XY平面
    np_scaled_vertices_2d = np_vertices_3d[:, :2] * scale_factors[:2]
  elif normal[0] >= normal[1] and normal[0] >= normal[2]:  # YZ平面
    np_scaled_vertices_2d = np_vertices_3d[:, 1:] * scale_factors[1:]
  else:  # XZ平面
    np_scaled_vertices_2d = np_vertices_3d[:, [0, 2]] * scale_factors[[0, 2]]

  return np.array(np_scaled_vertices_2d, dtype=np.float_).tolist()


def auto_triangulate(vertices_3d: list[list[float]]):
  """投影された2D座標でDelaunay三角形分割を行う"""
  scaled_vertices_2d = project_to_best_plane(vertices_3d)
  delaunay = Delaunay(scaled_vertices_2d)  # Delaunay 三角形分割を使用
  triangles: list[list[int]] = delaunay.simplices.tolist()

  # 結果としての三角形リスト
  filtered_triangles: list[list[int]] = []
  for triangle in triangles:
    # 三角形の重心がポリゴン内部にあるか
    centroid = calculate_centroid(triangle, scaled_vertices_2d)
    if is_point_in_polygon(centroid, scaled_vertices_2d):
      filtered_triangles.append(triangle)

  return filtered_triangles


def reorder_points(points: list[list[float]], clockwise=False):
  """
  Reorder the points of a polygon to ensure counter-clockwise or clockwise direction when viewed from above (along the Z-axis).

  Args:
  points (list[list[float]]): List of [x, y, z] coordinates for the vertices of the polygon.
  clockwise (bool): If True, reorder points to clockwise, otherwise counter-clockwise.

  Returns:
  list[list[float]]: Reordered list of polygon vertices.
  """

  polygon = Polygon(points)  # Use only x, y for Polygon
  is_currently_ccw = polygon.exterior.is_ccw

  # If the current order is not what is desired, reverse the points
  if (clockwise and is_currently_ccw) or (not clockwise and not is_currently_ccw):
    points = points[::-1]  # Reverse the list of points

  return points


def get_polygon_vertices_with_holes(
    polygon_vertices: list[list[float]],
    holes_vertices: list[list[list[float]]]
):
  all_vertices = polygon_vertices
  for holes_vertice in holes_vertices:
    roof_floor1_polygon = Polygon(all_vertices, holes=[holes_vertice])
    exterior_coords = [list(coord) for coord in roof_floor1_polygon.exterior.coords]
    holes_coords = [[list(coord) for coord in hole.coords] for hole in roof_floor1_polygon.interiors]
    all_vertices = exterior_coords + [coord for hole in holes_coords for coord in hole]

  return all_vertices


info = ObjInfo()


# 1階壁線
roof_floor1_wall_points = np.array(
    [[0, 0, 1.5], [10, 0, 1.7], [16, 6, 2], [10, 10, 1.7], [0, 10, 1.5]],
    dtype=np.float_
)
roof_floor1_height = np.min(roof_floor1_wall_points[:, 2])
roof_floor1_outer_ring_vertices = roof_floor1_wall_points.copy()
roof_floor1_outer_ring_vertices[:, 2] = roof_floor1_height

# 2階壁線
roof_floor2_walls_points: list[list[list[float]]] = np.array(
    [
        [[1, 1, 4.5], [3, 1, 4.7], [3, 3, 5], [2, 2.5, 4.7], [1, 3, 4.5]],
        [[6, 1, 4.5], [8, 1, 4.7], [8, 3, 5], [7, 2.5, 4.7], [6, 3, 4.5]]
    ],
    dtype=np.float_
)
roof_floor2_outer_rings_vertices = roof_floor2_walls_points.copy()
for roof_floor2_outer_ring_vertices in roof_floor2_outer_rings_vertices:
  roof_floor2_height = np.min(roof_floor2_outer_ring_vertices[:, 2])
  roof_floor2_outer_ring_vertices[:, 2] = roof_floor2_height
  info.append_faces(BldElementType.ROOF, [reorder_points(roof_floor2_outer_ring_vertices)])


# 穴のある多角形データ
roof_floor1_outer_ring_xy = np.array(roof_floor1_wall_points[:, :2], dtype=np.float_)  # 外部の境界
roof_floor1_holes_xy = [
    roof_floor2_outer_rings_vertice[:, :2]
    for roof_floor2_outer_rings_vertice in roof_floor2_outer_rings_vertices
]  # 内部の境界（穴）
roof_floor1_holes_vertices = [
    np.hstack([hole, np.full((hole.shape[0], 1), roof_floor1_height)]) for hole in roof_floor1_holes_xy
]

roof_floor1_vertices = get_polygon_vertices_with_holes(roof_floor1_outer_ring_vertices, roof_floor1_holes_vertices)
info.append_faces(BldElementType.ROOF, [reorder_points(roof_floor1_vertices)])


# 壁の生成
for roof_floor2_outer_ring_vertices in roof_floor2_outer_rings_vertices:
  for i in range(len(roof_floor2_outer_ring_vertices)):
    next_i = (i + 1) % len(roof_floor2_outer_ring_vertices)
    wall_vertices = [
        roof_floor2_outer_ring_vertices[i].tolist(),
        roof_floor2_outer_ring_vertices[next_i].tolist(),
        np.append(roof_floor2_outer_ring_vertices[next_i][:2], roof_floor1_height).tolist(),
        np.append(roof_floor2_outer_ring_vertices[i][:2], roof_floor1_height).tolist(),
    ]
    info.append_faces(BldElementType.WALL, [reorder_points(wall_vertices)])

info.write_file(file_path="test_roof_for_house_model_correct.obj")
