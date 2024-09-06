from typing import Union
import numpy as np
import open3d as o3d

from numpy.typing import NDArray
from shapely.geometry import Point, Polygon
from scipy.spatial import Delaunay

from src.thirdparty.plateaupy.thirdparty.earcutpython.earcut import earcut
from src.createmodel.housemodeling.house_model import HouseModel


class PointIndexFinder:
  def __init__(self, all_vertices):
    self.point_index_pair: dict[tuple[int, int, int], int] = {}
    for idx, point in enumerate(all_vertices):
      point_tuple = tuple(point)  # NumPy配列からタプルへ変換
      if point_tuple not in self.point_index_pair:
        self.point_index_pair[point_tuple] = idx

  def find(self, vertice: list[int]) -> Union[int, None]:
    point_tuple = tuple(vertice)
    return self.point_index_pair[point_tuple]


class RoofFloorObject:
  def __init__(
      self,
      vertices: NDArray[np.float_],
      vertices_3d: NDArray[np.float_],
      height: int,
      mesh
  ):
    self.vertices = vertices
    self.vertices_3d = vertices_3d
    self.height = height
    self.mesh = mesh


def create_mesh(vertices, faces):
  mesh = o3d.geometry.TriangleMesh()
  mesh.vertices = o3d.utility.Vector3dVector(vertices)
  mesh.triangles = o3d.utility.Vector3iVector(faces)
  mesh.compute_vertex_normals()
  return mesh


def prepare_earcut_data(floor_vertices, holes):
  """フロアと複数の穴の頂点データを準備してEarcutで使用する形に変換"""
  all_vertices = np.vstack([floor_vertices] + [hole[::-1] for hole in holes])  # 穴の順序を反時計回りにする
  flattened = all_vertices.flatten().tolist()
  hole_indices = [len(floor_vertices)]  # 最初の穴の開始インデックス
  current_index = len(floor_vertices)
  for hole in holes:
    current_index += len(hole)
    hole_indices.append(current_index)
  return flattened, hole_indices[:-1]  # 最後のインデックスは不要


def calculate_centroid(triangle, vertices):
  """Calculate the centroid of a triangle given by indices of its vertices."""
  vertex_coords = np.array([vertices[idx] for idx in triangle])
  centroid = np.mean(vertex_coords, axis=0)
  return centroid


def is_point_in_polygon(
    point: list[float],
    polygon: list[list[float]],
    holes: list[list[list[float]]] = []
):
  """点が多角形内にあるかどうかを判断する簡易的な方法"""
  poly = Polygon(polygon, holes)
  return poly.contains(Point(point))


def calculate_normal(vertices):
  """頂点から平面の法線を計算"""
  v1, v2, v3 = vertices[:3]
  return np.cross(v2 - v1, v3 - v1)


def project_to_best_plane(vertices_3d):
  """法線に基づいて最適な平面に頂点を投影し、距離歪みを補正"""
  normal = calculate_normal(vertices_3d)
  normal_abs = np.abs(normal / np.linalg.norm(normal))

  # 距離歪みを最小化するためのスケーリング係数
  scale_factors = np.sqrt(1 - normal_abs**2)

  if normal[2] >= normal[1] and normal[2] >= normal[0]:  # XY平面
    scaled_vertices_2d = vertices_3d[:, :2] * scale_factors[:2]
  elif normal[0] >= normal[1] and normal[0] >= normal[2]:  # YZ平面
    scaled_vertices_2d = vertices_3d[:, 1:] * scale_factors[1:]
  else:  # XZ平面
    scaled_vertices_2d = vertices_3d[:, [0, 2]] * scale_factors[[0, 2]]

  return scaled_vertices_2d


def auto_triangulate(vertices_3d):
  """投影された2D座標でDelaunay三角形分割を行う"""
  scaled_vertices_2d = project_to_best_plane(vertices_3d)
  delaunay = Delaunay(scaled_vertices_2d)  # Delaunay 三角形分割を使用
  triangles = delaunay.simplices

  # 結果としての三角形リスト
  filtered_triangles = []
  for triangle in triangles:
    # 三角形の重心がポリゴン内部にあるか
    centroid = calculate_centroid(triangle, scaled_vertices_2d)
    if is_point_in_polygon(centroid, scaled_vertices_2d):
      filtered_triangles.append(triangle)

  return filtered_triangles


poligon_points_list: list[list[list[float]]] = []

# 1階壁線
roof_floor1_vertices = np.array([[0, 0, 1.5], [10, 0, 1.7], [16, 6, 2], [10, 10, 1.7], [0, 10, 1.5]])
roof_floor1_height = np.min(roof_floor1_vertices[:, 2])
roof_floor1_polygon = roof_floor1_vertices.copy()
roof_floor1_polygon[:, 2] = roof_floor1_height
poligon_points_list.append(roof_floor1_polygon.tolist())

# 2階壁線
roof_floor2_walls: list[list[list[float]]] = [
    [[1, 1, 4.5], [3, 1, 4.7], [3, 3, 5], [2, 2.5, 4.7], [1, 3, 4.5]],
    [[6, 1, 4.5], [8, 1, 4.7], [8, 3, 5], [7, 2.5, 4.7], [6, 3, 4.5]]
]

roof_floor2_objects: list[RoofFloorObject] = []
for roof_floor2_wall in roof_floor2_walls:
  roof_floor2_vertices = np.array(roof_floor2_wall)
  roof_floor2_height = np.min(roof_floor2_vertices[:, 2])
  roof_floor2_vertices_3d = np.hstack([
      roof_floor2_vertices[:, :2], np.full((len(roof_floor2_vertices), 1), roof_floor2_height)
  ])
  poligon_points_list.append(roof_floor2_vertices_3d.tolist())
  triangles_indices_roof_floor2 = auto_triangulate(roof_floor2_vertices_3d)

  roof_floor2_mesh = create_mesh(roof_floor2_vertices_3d, triangles_indices_roof_floor2)
  roof_floor2_objects.append(RoofFloorObject(roof_floor2_vertices, roof_floor2_vertices_3d, roof_floor2_height, roof_floor2_mesh))


# 穴のある多角形データ
outer_ring = np.array(roof_floor1_vertices[:, :2])  # 外部の境界
holes_2d = [roof_floor2_object.vertices[:, :2] for roof_floor2_object in roof_floor2_objects]  # 内部の境界（穴）
holes_3d = [np.hstack([hole, np.full((hole.shape[0], 1), roof_floor1_height)]) for hole in holes_2d]

# 穴の頂点も含めてDelaunay分割
vertices_2d = np.vstack([outer_ring, *holes_2d])
vertices_3d = np.vstack([roof_floor1_polygon, *holes_3d])
delaunay = Delaunay(vertices_2d)

# 穴の内部にある三角形をフィルタリング
triangles = []
for simplex in delaunay.simplices:
  centroid = np.mean(vertices_2d[simplex], axis=0)
  if is_point_in_polygon(centroid, outer_ring, holes_2d):
    triangles.append(simplex)


# 1階の床のメッシュを生成
roof_floor1_mesh = create_mesh(vertices_3d, triangles)

# 壁の生成
roof_meshes = [roof_floor1_mesh, *[roof_floor2_object.mesh for roof_floor2_object in roof_floor2_objects]]

for roof_floor2_object in roof_floor2_objects:
  for i in range(len(roof_floor2_vertices)):
    next_i = (i + 1) % len(roof_floor2_vertices)
    wall_vertices = [
        roof_floor2_object.vertices_3d[i],
        roof_floor2_object.vertices_3d[next_i],
        np.append(roof_floor2_object.vertices_3d[next_i][:2], roof_floor1_height),
        np.append(roof_floor2_object.vertices_3d[i][:2], roof_floor1_height),
    ]
    poligon_points_list.append(wall_vertices)
    wall_faces = [[0, 1, 2], [0, 2, 3]]
    wall_mesh = create_mesh(np.array(wall_vertices), wall_faces)
    roof_meshes.append(wall_mesh)


# 全てのメッシュを合体
combined_roof_mesh = roof_meshes[0]
for mesh in roof_meshes[1:]:
  combined_roof_mesh += mesh

# 正解な屋根形状
o3d.io.write_triangle_mesh("test_roof_for_house_model_correct.obj", combined_roof_mesh, write_vertex_normals=True)
