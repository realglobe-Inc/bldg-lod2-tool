import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon
from shapely.geometry import Point as GeoPoint

from ..roof_layer_info import RoofLayerInfo
from ..model_surface_creation.utils.geometry import Point


def is_splitable_poligon(
    vertices: list[Point],
    polygon: list[int],
    layer_class: npt.NDArray[np.int_],
):
  """
  ポリゴン分割可能か確認。

  Args:
    vertices (list[Point]): 頂点リスト
    polygon (list[int]): ポリゴンの頂点番号リスト
    layer_class: (npt.NDArray[np.int_]) DSM点群の画像座標 (i,j) 二次元アレイに壁点を起点としてクラスタリングした屋根のレイヤー番号を記録したもの

  Returns:
    tuple[int, int]: 各頂点の(i, j)座標のリスト
  """
  if len(polygon) <= 4:
    return False

  layer_number_point_ijs_pair = get_layer_number_point_ijs_pair(vertices, polygon, layer_class)
  majority_layer_number = get_majority_layer_number(layer_number_point_ijs_pair)
  if majority_layer_number == RoofLayerInfo.NOISE_POINT:
    return False

  noise_ijs = layer_number_point_ijs_pair.get(RoofLayerInfo.NOISE_POINT) or []
  noise_count = len(noise_ijs)
  total_count = sum(len(v) for v in layer_number_point_ijs_pair.values()) - noise_count
  if total_count < 30:
    return False

  majority_layer_count = len(layer_number_point_ijs_pair[majority_layer_number])
  majority_layer_rate = majority_layer_count / total_count
  if total_count == 0 or majority_layer_rate > 0.85:
    return False

  return True


def point_id_to_ij(vertices: list[Point], point_id: list[int]) -> tuple[int, int]:
  """
  頂点IDを(i, j)座標に変換する。

  Args:
    vertices (list[Point]): 頂点リスト
    point_id (int): 頂点ID

  Returns:
    tuple[int, int]: 各頂点の(i, j)座標のリスト
  """
  return (int(vertices[point_id].x), int(vertices[point_id].y))


def get_layer_number_point_ijs_pair(vertices: list[Point], polygon: list[int], layer_class: npt.NDArray[np.int_]):
  """
  ポリゴン内のレイヤー番号と対応するポイントを取得する。

  Args:
    vertices (list[Point]): 頂点リスト
    polygon (list[int]): ポリゴンの頂点番号リスト
    layer_class: (npt.NDArray[np.int_]) DSM点群の画像座標 (i,j) 二次元アレイに壁点を起点としてクラスタリングした屋根のレイヤー番号を記録したもの

  Returns:
    dict[int, list[tuple[int, int]]]: レイヤー番号とそのポイント(i, j)のペアを含む辞書
  """

  polygon_ijs = [point_id_to_ij(vertices, point_id) for point_id in polygon]
  height, width = layer_class.shape
  poly = Polygon(polygon_ijs)
  layer_number_point_ijs_pair: dict[int, list[tuple[int, int]]] = {}
  for i in range(height):
    for j in range(width):
      is_inside_polygon = poly.contains(GeoPoint(i, j))
      if is_inside_polygon:
        layer_number = layer_class[i, j]

        if layer_number_point_ijs_pair.get(layer_number) is None:
          layer_number_point_ijs_pair[layer_number] = []

        layer_number_point_ijs_pair[layer_number].append((i, j))
  return layer_number_point_ijs_pair


def get_majority_layer_number(layer_number_point_ijs_pair: dict[int, list[tuple[int, int]]]):
  """
  ポリゴン内で最も多く出現するレイヤー番号を取得する。

  Args:
    layer_number_point_ijs_pair (dict[int, list[tuple[int, int]]]): レイヤー番号とそのポイント(i, j)のペアを含む辞書

  Returns:
    int: ポリゴン内で最も多く出現するレイヤー番号
  """
  layer_count_max = 0
  majority_layer_number = RoofLayerInfo.NOISE_POINT
  for layer_number, layer_points_ij in layer_number_point_ijs_pair.items():
    layer_count = len(layer_points_ij)

    if layer_count > layer_count_max and layer_number != RoofLayerInfo.NOISE_POINT:
      layer_count_max = layer_count
      majority_layer_number = layer_number

  return majority_layer_number


def ensure_counterclockwise(vertices: list[Point], polygon: list[int]):
  """
  ポリゴンの頂点が反時計回りになるようにする

  Args:
    vertices (list[Point]): 頂点リスト
    polygon (list[int]): ポリゴンの頂点番号リスト

  Returns:
    list[int]: 反時計回りにしたポリゴン
  """
  poly = Polygon([
      (vertices[point_id].x, vertices[point_id].y)
      for point_id in polygon
  ])
  if not poly.exterior.is_ccw:
      # 時計回りなら反転させる
    return polygon[::-1]

  return polygon


def calculate_angle_between_points(vertices: list[Point], polygon: list[int], point_id: int):
  """
  ポリゴンの指定した頂点を基準に前後の頂点との角度を計算する

  Args:
    vertices (list[Point]): 頂点リスト
    polygon (list[int]): ポリゴンの頂点番号リスト
    point_id (int): 基準となる頂点のインデックス

  Returns:
    float: 頂点の角度（度数法）
  """
  prev_point_id = -1
  next_point_id = -1
  for list_index, point_id_in_polygon in enumerate(polygon):
    if point_id_in_polygon == point_id:
      prev_list_index = len(polygon) - 1 if (list_index - 1) == -1 else list_index - 1
      next_list_index = 0 if (list_index + 1) == len(polygon) else list_index + 1

      prev_point_id = polygon[prev_list_index]
      next_point_id = polygon[next_list_index]

  # 現在の頂点
  current_point = vertices[point_id]
  # 前の頂点
  prev_point = vertices[prev_point_id]
  # 次の頂点
  next_point = vertices[next_point_id]

  # 前の頂点と次の頂点の傾きをそれぞれ計算
  prev_slope = calculate_slope(current_point, prev_point)
  next_slope = calculate_slope(current_point, next_point)

  # 前の頂点の角度 - 次の頂点の角度
  angle = prev_slope - next_slope

  # 角度が負の場合は360度を足す
  if angle < 0:
    angle += 360

  return angle


def calculate_slope(point1: Point, point2: Point) -> float:
  """
  2点間の傾きを計算する

  Args:
    point1 (Point): 頂点1
    point2 (Point): 頂点2

  Returns:
    float: i軸に対する傾き（角度）
  """
  dx = point2.x - point1.x
  dy = point2.y - point1.y
  return np.arctan2(dy, dx) * 180 / np.pi  # ラジアンから度に変換


def bresenham_line(ij_1: tuple[int, int], ij_2: tuple[int, int]):
  """
  Bresenhamのアルゴリズムを使って2点間の直線を描画する。

  Args:
    ij_1 (tuple[int, int]): 視点のピクセル座標
    ij_2 (tuple[int, int]): 終点のピクセル座標

  Returns:
    list[tuple[int, int]]: 線上のすべてのピクセル座標
  """
  i0, j0 = ij_1
  i1, j1 = ij_2
  pixels: list[tuple[int, int]] = []
  dx = abs(i1 - i0)
  dy = abs(j1 - j0)
  sx = 1 if i0 < i1 else -1
  sy = 1 if j0 < j1 else -1
  err = dx - dy

  while True:
    pixels.append((i0, j0))
    if i0 == i1 and j0 == j1:
      break
    e2 = 2 * err
    if e2 > -dy:
      err -= dy
      i0 += sx
    if e2 < dx:
      err += dx
      j0 += sy

  return pixels


def find_vertices_with_angle_over_200(vertices: list[Point], polygon: list[int]):
  """
  ポリゴンの内部から見て200度以上の角度を持つ頂点を探す
  Args:
    vertices (list[Point]): 頂点リスト
    polygon (list[int]): ポリゴンを構成する頂点のインデックスリスト

  Returns:
    list[int]: 200度以上の角度を持つ頂点のインデックスのリスト
  """

  vertices_with_large_angles: list[int] = []
  counter_clockwised_polygon = ensure_counterclockwise(vertices, polygon)
  for point_id in counter_clockwised_polygon:
    angle = calculate_angle_between_points(vertices, counter_clockwised_polygon, point_id)
    if angle > 200:
      vertices_with_large_angles.append(point_id)

  return vertices_with_large_angles


def get_point_ij_edge_point_id_start_end_pair(vertices: list[Point], polygon: list[int]):
  """
  ポリゴンのすべての辺のピクセル座標とそれに対応するエッジのペアを取得。

  Args:
    vertices (list[Point]): 頂点リスト
    polygon (list[int]): ポリゴンを構成する頂点のインデックスリスト

  Returns:
    dict[tuple[int, int], tuple[int, int]]: ポリゴンのすべての辺のピクセル座標とそれに対応するエッジのペア
  """

  point_ij_edge_point_id_start_end_pair: dict[tuple[int, int], tuple[int, int]] = {}

  for index, point_id in enumerate(polygon):
    # 頂点iと頂点i+1を結ぶ線を引く（最後の頂点は最初の頂点と結ぶ）
    current_ij = point_id_to_ij(vertices, point_id)
    next_index = index + 1 if (index + 1) != len(polygon) else 0
    next_point_id = polygon[next_index]
    next_ij = point_id_to_ij(vertices, next_point_id)

    pixle_ijs = bresenham_line(current_ij, next_ij)
    between_current_next_ijs = pixle_ijs[1:-1]

    # point_id の座標
    point_ij_edge_point_id_start_end_pair[current_ij] = (point_id, point_id)

    # point_id と next_point_id の間のある点の座標
    for between_current_next_ij in between_current_next_ijs:
      point_ij_edge_point_id_start_end_pair[between_current_next_ij] = (point_id, next_point_id)

  return point_ij_edge_point_id_start_end_pair


def get_splitable_point_on_poligon(
    vertices: list[Point],
    polygon: list[int],
    point_ij_edge_point_id_start_end_pair: dict[tuple[int, int], tuple[int, int]],
):
  """
  ポリゴン分割できる開始線の開始頂点IDと終了点の座標(i,j)リストのペアを出す。

  Args:
    vertices (list[Point]): 頂点リスト
    polygon (list[int]): ポリゴンを構成する頂点のインデックスリスト
    point_ij_edge_point_id_start_end_pair (dict[tuple[int, int], tuple[int, int]]): ポリゴンのすべての辺のピクセル座標とそれに対応するエッジのペア

  Returns:
    dict[int, list[tuple[int, int]]]: ポリゴン分割できる開始線の開始頂点IDと終了点の座標(i,j)リストのペア
    dict[tuple[int, int], tuple[int, int]]: ポリゴンのすべての辺のピクセル座標
  """

  start_point_id_available_end_point_ijs_pair: dict[int, list[tuple[int, int]]] = {}

  start_point_ids = find_vertices_with_angle_over_200(vertices, polygon)

  polygon_ijs = [point_id_to_ij(vertices, point_id) for point_id in polygon]
  poly = Polygon(polygon_ijs)

  for index, current_start_point_id in enumerate(start_point_ids):
    start_point_ids_length = len(start_point_ids)
    prev_start_point_id = start_point_ids[index - 1]
    next_start_point_id = start_point_ids[(index + 1) % start_point_ids_length]

    prev_start_point_ij, current_start_point_ij, next_start_point_ij = [
        point_id_to_ij(vertices, point_id) for point_id
        in [prev_start_point_id, current_start_point_id, next_start_point_id]
    ]

    available_end_point_ijs: list[tuple[int, int]] = []
    for end_point_ij, _ in point_ij_edge_point_id_start_end_pair.items():
      prev_polygon_line = bresenham_line(current_start_point_ij, prev_start_point_ij)
      next_polygon_line = bresenham_line(current_start_point_ij, next_start_point_ij)
      if end_point_ij in prev_polygon_line:
        continue

      if end_point_ij in next_polygon_line:
        continue

      line_points = bresenham_line(current_start_point_ij, end_point_ij)
      line_points_without_poligon_outer_points = [
          line_point for line_point in line_points
          if line_point not in point_ij_edge_point_id_start_end_pair
      ]

      can_select_point = True
      for line_point in line_points_without_poligon_outer_points:
        if poly.contains(GeoPoint(line_point[0], line_point[1])) is False:
          can_select_point = False

      if len(line_points_without_poligon_outer_points) == 0:
        can_select_point = False

      if can_select_point:
        available_end_point_ijs.append(end_point_ij)

    start_point_id_available_end_point_ijs_pair[current_start_point_id] = available_end_point_ijs

  return start_point_id_available_end_point_ijs_pair


def split_polygon(
    vertices: list[Point],
    polygon: list[int],
    point_id1: int,
    point_id2: int,
) -> tuple[list[int], list[int]]:
  """
  ポリゴンの頂点リストを任意の2点で分割して2つのポリゴンを生成する。

  Args:
      vertices (list[Point]): 頂点リスト
      polygon (list[int]): ポリゴンの頂点番号リスト
      point_id1 (int): 分割する最初の点のID
      point_id2 (int): 分割する2番目の点のID

  Returns:
      tuple: 2つの分割されたポリゴンの頂点番号リスト
  """
  # インデックスを探す
  index1 = polygon.index(point_id1)
  index2 = polygon.index(point_id2)

  # index1 が index2 より小さくなるように調整
  if index1 > index2:
    index1, index2 = index2, index1

  # ポリゴンを2つに分割
  polygon1 = polygon[index1:index2 + 1]  # 最初の部分
  polygon2 = polygon[index2:] + polygon[:index1 + 1]  # 2つ目の部分

  return [ensure_counterclockwise(vertices, polygon) for polygon in [polygon1, polygon2]]


def get_splited_poligons_score(
    vertices: list[Point],
    splited_polygons: list[list[int]],
    layer_class: npt.NDArray[np.int_],
):
  """
  分割されたポリゴンがどれだけ綺麗に分割されたかのスコアを計算。
  - ノイズ処理された点は計算から除外
  - 分割されたポリゴンの内部に入っている DSM 点の数が 5 以下の場合 0点
  - SUM（「クラス k の点の数」 / 「ポリゴン内部に入っている DSM 点の数」）^2

  Args:
    vertices (list[Point]): 頂点リスト
    splited_polygons (list[list[int]]): 分割された二つのポリゴンの頂点番号リスト
    layer_class: (npt.NDArray[np.int_]) DSM点群の画像座標 (i,j) 二次元アレイに壁点を起点としてクラスタリングした屋根のレイヤー番号を記録したもの

  Returns:
    tuple[int, int]: 各頂点の(i, j)座標のリスト
  """

  total_score = float(0)
  for splited_polygon in splited_polygons:
    # 頂点が二つ以下の場合、ポリゴンではない場合 0点
    if len(splited_polygon) <= 2:
      return float(0)

    layer_number_point_ijs_pair = get_layer_number_point_ijs_pair(vertices, splited_polygon, layer_class)
    total_point_count = 0
    for layer_number, point_ijs in layer_number_point_ijs_pair.items():
      if layer_number >= 0:
        total_point_count += len(point_ijs)

    # 分割されたポリゴンの内部に入っている DSM 点の数が 5 以下の場合 0点
    if total_point_count <= 5:
      return float(0)

    # SUM（「クラス k の点の数」 / 「ポリゴン内部に入っている DSM 点の数」）^2
    for layer_number, point_ijs in layer_number_point_ijs_pair.items():
      if layer_number >= 0:
        total_score += ((len(point_ijs) / total_point_count) ** 2)

  return total_score
