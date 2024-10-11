import os
from typing import Union

import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import Point as GeoPoint
import cv2

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
    tuple[float, float]: 各頂点の(i, j)座標のリスト
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

  has_angle_over_200 = False
  counter_clockwised_polygon = ensure_counterclockwise(vertices, polygon)
  for point_id in counter_clockwised_polygon:
    angle = calculate_angle_between_points(vertices, counter_clockwised_polygon, point_id)
    if angle > 200:
      has_angle_over_200 = True

  return has_angle_over_200


def point_id_to_ij(vertices: list[Point], point_id: list[int]) -> tuple[float, float]:
  """
  頂点IDを(i, j)座標に変換する。

  Args:
    vertices (list[Point]): 頂点リスト
    point_id (int): 頂点ID

  Returns:
    tuple[float, float]: 各頂点の(i, j)座標のリスト
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
    dict[int, list[tuple[float, float]]]: レイヤー番号とそのポイント(i, j)のペアを含む辞書
  """

  polygon_ijs = [point_id_to_ij(vertices, point_id) for point_id in polygon]
  height, width = layer_class.shape
  poly = Polygon(polygon_ijs)
  layer_number_point_ijs_pair: dict[int, list[tuple[float, float]]] = {}
  for i in range(height):
    for j in range(width):
      is_inside_polygon = poly.contains(GeoPoint(i, j))
      if is_inside_polygon:
        layer_number = layer_class[i, j]

        if layer_number_point_ijs_pair.get(layer_number) is None:
          layer_number_point_ijs_pair[layer_number] = []

        layer_number_point_ijs_pair[layer_number].append((i, j))
  return layer_number_point_ijs_pair


def get_polygon_inner_point_ijs(vertices: list[Point], polygon: list[int], layer_class: npt.NDArray[np.int_]):
  polygon_ijs = [point_id_to_ij(vertices, point_id) for point_id in polygon]
  height, width = layer_class.shape
  poly = Polygon(polygon_ijs)
  polygon_inner_point_ijs: list[tuple[float, float]] = []
  for i in range(height):
    for j in range(width):
      is_inside_polygon = poly.contains(GeoPoint(i, j))
      if is_inside_polygon:
        polygon_inner_point_ijs.append((i, j))

  return polygon_inner_point_ijs


def get_majority_layer_number(layer_number_point_ijs_pair: dict[int, list[tuple[float, float]]]):
  """
  ポリゴン内で最も多く出現するレイヤー番号を取得する。

  Args:
    layer_number_point_ijs_pair (dict[int, list[tuple[float, float]]]): レイヤー番号とそのポイント(i, j)のペアを含む辞書

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


def bresenham_line(ij_1: tuple[float, float], ij_2: tuple[float, float]):
  """
  Bresenhamのアルゴリズムを使って2点間の直線を描画する。

  Args:
    ij_1 (tuple[float, float]): 視点のピクセル座標
    ij_2 (tuple[float, float]): 終点のピクセル座標

  Returns:
    list[tuple[float, float]]: 線上のすべてのピクセル座標
  """
  i0, j0 = ij_1
  i1, j1 = ij_2
  pixels: list[tuple[float, float]] = []
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


def get_new_intersection_polygon_ijs(
    vertices: list[Point],
    polygon: list[int],
    roof_layer_info: RoofLayerInfo,
    debug_mode: bool = False,
):
  """


  Args:
    vertices (list[Point]): 頂点リスト
    polygon (list[int]): ポリゴンの頂点番号リスト
    roof_layer_info: RoofLayerInfo 計算結果

  Returns:
    tuple[float, float]: 各頂点の(i, j)座標のリスト
  """
  # rgb_image = roof_layer_info.rgb_image.copy()
  layer_class = roof_layer_info.layer_class.copy()

  layer_number_point_ijs_pair = get_layer_number_point_ijs_pair(vertices, polygon, layer_class)

  layer_numbers = [
      layer_number for layer_number in layer_number_point_ijs_pair.keys()
      if layer_number >= 0
  ]

  intersection_polygon_ijs_list: list[list[tuple[float, float]]] = []
  polygon_ijs = [point_id_to_ij(vertices, point_id) for point_id in polygon]
  for layer_number in layer_numbers:
    # 屋根のレイヤー番号別にポリゴン化
    layer_outline_ijs_list = roof_layer_info.layer_number_layer_outline_polygons_list_pair[layer_number]
    for layer_outline_ijs in layer_outline_ijs_list:
      # 屋根のレイヤーのポリゴンと屋根線内部ポリゴンが被っている領域のポリゴン
      intersection_polys = get_intersection_of_polygons(polygon_ijs, layer_outline_ijs)
      origin_intersection_polygon_ijs: list[list[tuple[float, float]]] = []
      for intersection_poly in intersection_polys:
        origin_intersection_polygon_ijs: list[tuple[float, float]] = [
            coord for coord in intersection_poly.exterior.coords[:-1]
        ]

      new_intersection_polygon_ijs: list[list[tuple[float, float]]] = [
          (round(coord[0]), round(coord[1])) for coord in origin_intersection_polygon_ijs
      ]
      new_intersection_polygon_ijs = remove_same_vertices_on_polygon(new_intersection_polygon_ijs)
      if len(new_intersection_polygon_ijs) >= 3:
        if Polygon(new_intersection_polygon_ijs).is_valid:
          intersection_polygon_ijs_list.append(new_intersection_polygon_ijs)
        else:
          # 座標移動によって不正ポリゴンになる可能性がある場合は、移動した座標をもとに戻す
          intersection_polygon_ijs_list.append(origin_intersection_polygon_ijs)

  removed_small_polygon_ijs_list = [
      intersection_polygon_ijs for intersection_polygon_ijs in intersection_polygon_ijs_list
      if Polygon(intersection_polygon_ijs).area > 15
  ]

  other_polygon: Union[MultiPolygon, Polygon] = Polygon(polygon_ijs)
  for removed_small_polygon_ijs in removed_small_polygon_ijs_list:
    removed_small_poly = Polygon(removed_small_polygon_ijs)
    other_polygon = other_polygon.difference(removed_small_poly)

  splited_polygon_ijs_list = intersection_polygon_ijs_list
  if not other_polygon.is_empty:
    if isinstance(other_polygon, MultiPolygon):
      for poly in other_polygon:
        polygon = [coord for coord in poly.exterior.coords[:-1]]
        splited_polygon_ijs_list.append(polygon)
    elif isinstance(other_polygon, Polygon):
      polygon = [coord for coord in other_polygon.exterior.coords[:-1]]
      splited_polygon_ijs_list.append(polygon)

  merged_polygon_ijs_list = merge_polygon_vertices(polygon_ijs, splited_polygon_ijs_list)
  filterd_polygon_ijs_list = []
  for merged_polygon_ijs in merged_polygon_ijs_list:
    if is_polygon_inside_polygon(polygon_ijs, merged_polygon_ijs):
      print('inside')
      filterd_polygon_ijs_list.append(merged_polygon_ijs)
    else:
      print('outside')
      filterd_polygon_ijs_list.append(merged_polygon_ijs)

  if debug_mode:
    height, width = roof_layer_info.rgb_image.shape[:2]
    image_layer_splited_polygons_image = np.full((height, width, 3), 255, dtype=np.uint8)

    polygons_np = [
        np.array(merged_polygon_ijs, np.int32)[:, ::-1].reshape((-1, 1, 2))
        for merged_polygon_ijs in merged_polygon_ijs_list
    ]
    edge_color = roof_layer_info.get_color(RoofLayerInfo.ROOF_LINE_POINT)
    cv2.polylines(image_layer_splited_polygons_image, polygons_np, isClosed=True, color=edge_color, thickness=1)

    point_color = roof_layer_info.get_color(RoofLayerInfo.ROOF_VERTICE_POINT)
    for filterd_polygon_ijs in filterd_polygon_ijs_list:
      for i, j in filterd_polygon_ijs:
        cv2.circle(image_layer_splited_polygons_image, (round(j), round(i)), 0, point_color, -1)

    image_layer_splited_polygons_path = os.path.join(roof_layer_info.debug_dir, 'layer_splited_polygons.png')
    cv2.imwrite(image_layer_splited_polygons_path, image_layer_splited_polygons_image)

  print(filterd_polygon_ijs_list)

  return filterd_polygon_ijs_list


def is_polygon_inside_polygon(polygon_large: list[tuple[float, float]], polygon_small: list[tuple[float, float]]):
  """
  ポリゴンのに他のポリゴン達が含まれているか確認する

  Args:
    polygon_large (list[tuple[float, float]]): ポリゴンの頂点番号リスト
    polygon_small (list[tuple[float, float]]): ポリゴンの頂点番号リスト

  Returns:
    bool: 各頂点の(i, j)座標のリスト
  """
  poly_small = Polygon(polygon_small)
  poly_large = Polygon(polygon_large)

  result = poly_small.difference(poly_large)
  return result.is_empty


def merge_polygon_vertices(
    origin_polygon_ijs: list[tuple[float, float]],
    splited_polygon_ijs_list: list[list[tuple[float, float]]],
):
  """
  分割されたポリゴンの頂点を元のポリゴンの頂点に合わせる(1px 範囲)

  Args:
    origin_polygon_ijs (list[tuple[float, float]]): 元のポリゴンの頂点番号リスト
    splited_polygon_ijs_list (list[list[tuple[float, float]]]): 分割されたポリゴン達の頂点番号リスト

  Returns:
    list[list[tuple[float, float]]]: ポリゴンの頂点に合わせて頂点を変更した分割されたポリゴン達の頂点番号リスト
  """

  merged_polygon_ijs_list: list[list[tuple[float, float]]] = []
  for origin_polygon_ijs in splited_polygon_ijs_list:
    merged_polygon_ijs: list[tuple[float, float]] = []
    for from_ij in origin_polygon_ijs:
      from_i, from_j = from_ij
      merged_ij: tuple[int, int] = (from_i, from_j)
      for to_i, to_j in origin_polygon_ijs:
        if (
            to_i - 1 <= from_i <= to_i + 1
            and to_j - 1 <= from_j <= to_j + 1
            and (to_i, to_j) != (from_i, from_j)
        ):
          merged_ij = (to_i, to_j)
          break

      merged_polygon_ijs.append(merged_ij)

    merged_polygon_ijs = remove_same_vertices_on_polygon(merged_polygon_ijs)
    if len(merged_polygon_ijs) >= 3:
      if Polygon(merged_polygon_ijs).is_valid:
        merged_polygon_ijs_list.append(merged_polygon_ijs)
      else:
        # 座標移動によって不正ポリゴンになる可能性がある場合は、移動した座標をもとに戻す
        merged_polygon_ijs_list.append(origin_polygon_ijs)

  return merged_polygon_ijs_list


def get_intersection_of_polygons(polygon1_ij: list[tuple[float, float]], polygon2_ij: list[tuple[float, float]]):
  """
  ポリゴン同士の交差している領域を求める

  Args:
    polygon1_ij (list[tuple[float, float]]): ポリゴンの頂点番号リスト
    polygon2_ij (list[tuple[float, float]]): ポリゴンの頂点番号リスト

  Returns:
    list[Polygon]: 交差している領域のポリゴンリスト
  """

  # 差分を取得
  polygon1 = Polygon(polygon1_ij)
  polygon2 = Polygon(polygon2_ij)

  intersection: Union[MultiPolygon, Polygon] = polygon1.intersection(polygon2)
  intersection_polys: list[Polygon] = []
  if not intersection.is_empty:
    if isinstance(intersection, Polygon):
      intersection_polys = [intersection]
    elif isinstance(intersection, MultiPolygon):
      intersection_polys = [intersection_poly for intersection_poly in intersection]

  return intersection_polys


def is_integer_value(value: float) -> bool:
  """
  float 型の数値が整数（小数点以下が 0）であるかを確認する関数。
  """
  return value % 1 == 0


def remove_same_vertices_on_polygon(polygon_ijs: list[tuple[float, float]]):
  """
  ポリゴン内部に同じ点が連続にあるのを防ぐ

  Args:
    polygon_ij (list[tuple[float, float]]): ポリゴンの頂点番号リスト

  Returns:
    list[tuple[float, float]]: 交差している領域のポリゴンリスト
  """

  new_polygon_ijs = []
  for current_polygon_ij in polygon_ijs:
    if len(new_polygon_ijs) == 0:
      new_polygon_ijs.append(current_polygon_ij)
    else:
      prev_polygon_ij = new_polygon_ijs[-1]
      next_polygon_ij = new_polygon_ijs[0]

      if prev_polygon_ij != current_polygon_ij and next_polygon_ij != current_polygon_ij:
        new_polygon_ijs.append(current_polygon_ij)

  return new_polygon_ijs
