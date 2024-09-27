import numpy as np
from shapely.geometry import Polygon


def ensure_counterclockwise(polygon):
  """
  ポリゴンの頂点が反時計回りになるようにする
  Args:
      polygon (Polygon): ShapelyのPolygonオブジェクト

  Returns:
      Polygon: 反時計回りにしたポリゴン
  """
  if signed_area(polygon) < 0:
    # 時計回りなら反転させる
    return Polygon(polygon.exterior.coords[::-1])
  return polygon


def signed_area(polygon):
  """
  ポリゴンの符号付き面積を計算する関数
  面積が正であれば反時計回り、負であれば時計回り
  Args:
      polygon (Polygon): ShapelyのPolygonオブジェクト

  Returns:
      float: 符号付き面積
  """
  vertices = list(polygon.exterior.coords)
  n = len(vertices) - 1  # 最後の点は最初の点と同じなので除外

  area = 0.0
  for i in range(n):
    x1, y1 = vertices[i]
    x2, y2 = vertices[(i + 1) % n]
    area += (x2 - x1) * (y2 + y1)

  return area / 2.0


def calculate_slope(point1, point2):
  """
  2点間の傾きを計算する
  Args:
      point1, point2: 2点の座標 [x, y]

  Returns:
      float: i軸に対する傾き（角度）
  """
  dx = point2[0] - point1[0]
  dy = point2[1] - point1[1]
  return np.arctan2(dy, dx) * 180 / np.pi  # ラジアンから度に変換


def calculate_angle_between_points(polygon, index):
  """
  ポリゴンの指定した頂点を基準に前後の頂点との角度を計算する
  Args:
      polygon (Polygon): ShapelyのPolygonオブジェクト
      index (int): 基準となる頂点のインデックス

  Returns:
      float: 頂点の角度（度数法）
  """
  vertices = list(polygon.exterior.coords)
  num_vertices = len(vertices) - 1  # 最後の点は最初の点と同じなので除外

  # 基準となる頂点
  current_point = vertices[index]
  # 前の頂点
  prev_point = vertices[index - 1]
  # 次の頂点
  next_point = vertices[(index + 1) % num_vertices]

  # 前の頂点と次の頂点の傾きをそれぞれ計算
  prev_slope = calculate_slope(current_point, prev_point)
  next_slope = calculate_slope(current_point, next_point)

  # 次の頂点の角度 - 前の頂点の角度
  angle = next_slope - prev_slope

  # 角度が負の場合は360度を足す
  if angle < 0:
    angle += 360

  return angle


def find_vertices_with_angle_over_180(polygon):
  """
  ポリゴンの内部から見て180度以上の角度を持つ頂点を探す
  Args:
      polygon (Polygon): ShapelyのPolygonオブジェクト

  Returns:
      list[tuple[int, tuple[float, float], float]]: 
          180度以上の角度を持つ頂点のインデックス、座標、角度のリスト
  """
  polygon = ensure_counterclockwise(polygon)
  vertices = list(polygon.exterior.coords)
  num_vertices = len(vertices) - 1  # 最後の点は最初の点と同じなので除外
  vertices_with_large_angles = []

  for i in range(num_vertices):
    angle = calculate_angle_between_points(polygon, i)

    if angle > 180:
      vertices_with_large_angles.append((i, vertices[i], angle))

  return vertices_with_large_angles


# ShapelyのPolygonを使ってポリゴンを定義
polygon = Polygon([[0, 0], [4, 0], [4, 4], [2, 2], [0, 4]])

# 180度以上の角度を持つ頂点を取得
large_angle_vertices = find_vertices_with_angle_over_180(polygon)

# 結果を表示
if large_angle_vertices:
  print("180度以上の角度を持つ頂点:")
  for index, vertex, angle in large_angle_vertices:
    print(f"頂点 {index}: {vertex}, 角度: {angle:.2f}度")
else:
  print("180度以上の角度を持つ頂点はありません。")
