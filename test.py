import numpy as np


def calculate_slope(point1: list[float], point2: list[float]):
  """
  2点間の傾きを計算する
  Args:
    point1 (list[float]): 点1の座標 [x, y]
    point2 (list[float]): 点2の座標 [x, y]

  Returns:
    float: i軸に対する傾き（角度）
  """
  dx = point2[0] - point1[0]
  dy = point2[1] - point1[1]
  return np.arctan2(dy, dx) * 180 / np.pi  # ラジアンから度に変換


def calculate_angle_between_points(
    vertices_ij: list[list[float]],
    point_id: int,
    polygon: list[int],
):
  """
  ポリゴンの指定した頂点を基準に前後の頂点との角度を計算する
  Args:
    vertices_ij (list[list[float]]): 全頂点の座標リスト
    point_id (int): 基準となる頂点のインデックス
    polygon (list[int]): ポリゴンを構成する頂点のインデックスリスト

  Returns:
    float: 頂点の角度（度数法）
  """
  num_vertices = len(polygon)

  # 現在の頂点
  current_point = vertices_ij[polygon[point_id]]
  # 前の頂点
  prev_point = vertices_ij[polygon[point_id - 1]]
  # 次の頂点
  next_point = vertices_ij[polygon[(point_id + 1) % num_vertices]]

  # 前の頂点と次の頂点の傾きをそれぞれ計算
  prev_slope = calculate_slope(current_point, prev_point)
  next_slope = calculate_slope(current_point, next_point)

  # 次の頂点の角度 - 前の頂点の角度
  angle = next_slope - prev_slope

  # 角度が負の場合は360度を足す
  if angle < 0:
    angle += 360

  return angle


def find_vertices_with_angle_over_180(vertices_ij, polygon):
  """
  ポリゴンの内部から見て180度以上の角度を持つ頂点を探す
  Args:
    vertices_ij (list[list[float]]): 全頂点の座標リスト
    polygon (list[int]): ポリゴンを構成する頂点のインデックスリスト

  Returns:
    list[tuple[int, list[float], float]]: 
      180度以上の角度を持つ頂点のインデックス、座標、角度のリスト
  """
  vertices_with_large_angles = []

  for i in range(len(polygon)):
    # 各頂点に対して角度を計算
    angle = calculate_angle_between_points(vertices_ij, i, polygon)

    if angle > 180:
      vertices_with_large_angles.append((polygon[i], vertices_ij[polygon[i]], angle))

  return vertices_with_large_angles


# 頂点座標のリスト
vertices_ij = [[0, 0], [4, 0], [4, 4], [2, 2], [0, 4]]

# ポリゴンを構成する頂点のインデックスリスト
polygon = [0, 1, 2, 3, 4]

# 180度以上の角度を持つ頂点を取得
large_angle_vertices = find_vertices_with_angle_over_180(vertices_ij, polygon)

# 結果を表示
if large_angle_vertices:
  print("180度以上の角度を持つ頂点:")
  for index, vertex, angle in large_angle_vertices:
    print(f"頂点 {index}: {vertex}, 角度: {angle:.2f}度")
else:
  print("180度以上の角度を持つ頂点はありません。")
