from typing import Union
import numpy as np
import open3d as o3d

from numpy.typing import NDArray
from shapely.geometry import Polygon

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


def auto_triangulate(vertices):
  """ 自動的に頂点から三角形インデックスを生成する """
  flattened = vertices[:, :2].flatten()  # Z座標を無視
  triangles_indices = earcut.earcut(flattened, dim=2)
  return np.array(triangles_indices).reshape((-1, 3))


poligon_points_list: list[list[list[float]]] = []

# 1階壁線
floor1_vertices = np.array([[0, 0, 1.5], [10, 0, 1.7], [16, 6, 2], [10, 10, 1.7], [0, 10, 1.5]])
floor1_height = np.min(floor1_vertices[:, 2])
floor1_polygon = floor1_vertices.copy()
floor1_polygon[:, 2] = floor1_height
poligon_points_list.append(floor1_polygon.tolist())

# 2階壁線
floor2_walls: list[list[list[float]]] = [
    [[1, 1, 4.5], [3, 1, 4.7], [3, 3, 5], [2, 2.5, 4.7], [1, 3, 4.5]],
    # [[6, 1, 4.5], [8, 1, 4.7], [8, 3, 5], [7, 2.5, 4.7], [6, 3, 4.5]]
]

floor2_objects: list[RoofFloorObject] = []
for floor2_wall in floor2_walls:
  floor2_vertices = np.array(floor2_wall)
  floor2_height = np.min(floor2_vertices[:, 2])
  floor2_vertices_3d = np.hstack([
      floor2_vertices[:, :2], np.full((len(floor2_vertices), 1), floor2_height)
  ])
  poligon_points_list.append(floor2_vertices_3d.tolist())
  triangles_indices_floor2 = auto_triangulate(floor2_vertices)
  floor2_mesh = create_mesh(floor2_vertices_3d, triangles_indices_floor2)
  floor2_objects.append(RoofFloorObject(floor2_vertices, floor2_vertices_3d, floor2_height, floor2_mesh))

# Earcutで三角形化の準備
flattened, hole_indices = prepare_earcut_data(
    floor1_vertices[:, :2], [floor2_object.vertices[:, :2] for floor2_object in floor2_objects]
)
tri_indices = earcut.earcut(flattened, hole_indices, 2)
vertices_3d = np.hstack([np.array(flattened).reshape(-1, 2), np.full((len(flattened) // 2, 1), floor1_height)])
triangles = np.array(tri_indices).reshape((-1, 3))

# 1階の床のメッシュを生成
floor_mesh = create_mesh(vertices_3d, triangles)

# 壁の生成
meshes = [floor_mesh, *[floor2_object.mesh for floor2_object in floor2_objects]]

for floor2_object in floor2_objects:
  for i in range(len(floor2_vertices)):
    next_i = (i + 1) % len(floor2_vertices)
    wall_vertices = [
        floor2_object.vertices_3d[i],
        floor2_object.vertices_3d[next_i],
        np.append(floor2_object.vertices_3d[next_i][:2], floor1_height),
        np.append(floor2_object.vertices_3d[i][:2], floor1_height),
    ]
    poligon_points_list.append(wall_vertices)
    wall_faces = [[0, 1, 2], [0, 2, 3]]
    wall_mesh = create_mesh(np.array(wall_vertices), wall_faces)
    meshes.append(wall_mesh)


# 全てのメッシュを合体
combined_mesh = meshes[0]
for mesh in meshes[1:]:
  combined_mesh += mesh


# 全頂点を抽出
all_vertices = np.asarray(combined_mesh.vertices)
print("All Vertices:")
print(all_vertices)

# 全ポリゴン（三角形）を抽出
all_triangles = np.asarray(combined_mesh.triangles)
print("All Polygons (Triangles):")
print(all_triangles)

# 頂点の座標リストを出力
vertices_list = all_vertices.tolist()
print("Vertices List:")
print(vertices_list)

# 三角形のインデックスリストを出力
triangles_index_list = all_triangles.tolist()
print("Triangles Index List:")
print(triangles_index_list)

# 正解な屋根形状
o3d.io.write_triangle_mesh("test_roof_for_house_model_correct.obj", combined_mesh, write_vertex_normals=True)

point_index_finder = PointIndexFinder(all_vertices)

# 一致するインデックスを見つける
outer_polygon = [point_index_finder.find(floor1_vertice) for floor1_vertice in floor1_polygon]

inner_polygons: list[list[int]] = []
for poligon_points in poligon_points_list:
  inner_polygons.append([point_index_finder.find(poligon_point) for poligon_point in poligon_points])

# テスト結果は test_roof_for_house_model_result.obj
model = HouseModel(id='test_roof_for_house_model_result', shape=Polygon(floor1_vertices[:, :2]))
model.create_model_surface(
    point_cloud=all_vertices,
    points_xy=all_vertices[:, :2],
    inner_polygons=inner_polygons,
    outer_polygon=outer_polygon,
    ground_height=0,
    balcony_flags=[False for _ in all_triangles]
)

model.simplify(threshold=5)
model.rectify()
