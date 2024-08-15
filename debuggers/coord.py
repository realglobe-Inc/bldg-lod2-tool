import argparse

from pyproj import Transformer


def main():
  parser = argparse.ArgumentParser("")
  parser.add_argument("src_epsg", type=str, help="Source EPSG code : 6668 or 6669~6687")
  parser.add_argument("dst_epsg", type=str, help="Destination EPSG code : 6669~6687 or 6668")
  parser.add_argument("pos_a", type=float, help="Coordinate X or Latitute")
  parser.add_argument("pos_b", type=float, help="Coordinate Y or Longitude")
  args = parser.parse_args()

  # python coord.py 6668 6677 35.91760209932579 139.27930937708183
  # python coord.py 6677 6668 -9000 -50000

  transformer = Transformer.from_crs(f"epsg:{args.src_epsg}", f"epsg:{args.dst_epsg}", always_xy=True)
  # 6668(緯度経度座標)から、6669~6687(平面直角座標)へ
  if args.src_epsg == "6668":
    lat = args.pos_a
    log = args.pos_b
    x, y = transformer.transform(log, lat)
    print(x, y)

  # 6669~6687(平面直角座標)から、6668(緯度経度座標)へ
  else:
    x = args.pos_a
    y = args.pos_b
    lon, lat = transformer.transform(x, y)
    print(lat, lon)


if __name__ == "__main__":
  main()
