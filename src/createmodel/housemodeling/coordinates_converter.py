from typing import Final


class CoordConverterForCartesianAndImagePos:
  """画像座標と平面直角座標を変換するクラス
  """

  _grid_size: Final[float]
  _cartesian_coord_upper_left: Final[tuple[float, float]]

  def __init__(
      self,
      grid_size: float,
      cartesian_coord_upper_left: tuple[float, float],
  ) -> None:
    """コンストラクタ

    Args:
      grid_size(float): 点群の間隔(meter)
      cartesian_coord_upper_left(tuple[float,float]) 画像左上位置の平面直角座標(x, yの順)
    """
    self._grid_size = grid_size
    self._cartesian_coord_upper_left = cartesian_coord_upper_left

  def cartesian_point_to_image_point(self, geo_x: float, geo_y: float) -> tuple[int, int]:
    """画像座標から平面直角座標へ変換する
    Args:
      geo_x(float): 平面直角座標のx座標
      geo_y(float): 平面直角座標のy座標

    Returns:
      int: 画像のx座標(左を0とする)
      int: 画像のy座標(上を0とする)
    """

    left, upper = self._cartesian_coord_upper_left
    return (
        round((geo_x - left) / self._grid_size),
        round((upper - geo_y) / self._grid_size)
    )

  def image_point_to_cartesian_point(self, image_x: float, image_y: float) -> tuple[float, float]:
    """画像座標から平面直角座標へ変換する
    Args:
      image_x(float): 画像のx座標(左を0とする)
      image_y(float): 画像のy座標(上を0とする)

    Returns:
      float: 平面直角座標のx座標
      float: 平面直角座標のy座標
    """
    left, upper = self._cartesian_coord_upper_left
    return (
        left + image_x * self._grid_size,
        upper - image_y * self._grid_size,
    )
