import os
import argparse
import xml.etree.ElementTree as ET

import laspy
from pyproj import Transformer


def is_bbox_overlap(bbox1: tuple[float], bbox2: tuple[float]):
    """
    Check if two bounding boxes overlap.
    bbox1 and bbox2: (min_x, min_y, max_x, max_y)
    """
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])



def get_las_bbox(las_file: str):
    """
    Get the bounding box (min_x, min_y, max_x, max_y) of the LAS file.
    """
    try:
        with laspy.open(las_file) as f:
            header = f.header
            min_x: float = header.mins[0]
            max_x: float = header.maxs[0]
            min_y: float = header.mins[1]
            max_y: float = header.maxs[1]
            return (min_x, min_y, max_x, max_y)
    except Exception as e:
        print(f"Error reading {las_file}: {e}")
        return None


def search_dsm_files(bbox: tuple[float], directory: str):
    """
    Search for LAS files that overlap with the given bounding box.
    """
    matching_files: list[str] = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".las"):
                file_path = os.path.join(root, file)
                las_bbox: tuple[int] = get_las_bbox(file_path)
                if las_bbox and is_bbox_overlap(las_bbox, bbox):
                    matching_files.append(file_path)

    return matching_files


def to_cartesian_point(lat: float, lon: float, dst_epsg: int):
    """
    Convert latitude and longitude to the target Cartesian coordinate system.
    """
    try:
        transformer = Transformer.from_crs(f"epsg:6668", f"epsg:{dst_epsg}", always_xy=True)
        # 6668 (緯度経度座標) から、6677などの平面直角座標へ変換
        x, y = transformer.transform(float(lon), float(lat))
        return x, y
    except Exception as e:
        print(f"Error converting coordinates: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Search DSM files overlapping with CityGML bounding box")
    parser.add_argument("citygml_file", type=str, help="Path to the CityGML file")
    parser.add_argument("dsm_dir", type=str, help="Directory containing DSM (LAS) files")
    parser.add_argument("dsm_epsg", type=int, help="EPSG code for DSM (LAS) file")
    args = parser.parse_args()

    # python search_dsm_file_by_city_gml.py \
    #   ~/lod2_data/kawazaki/08_CityGML/53392535_bldg_6697_op.gml \
    #   /lod2_data/kawazaki/04_DSM_RGB 6677

    # XMLファイルをパース
    try:
        tree = ET.parse(args.citygml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return

    # 名前空間の定義
    ns = {"gml": "http://www.opengis.net/gml"}

    # lowerCornerとupperCornerを取得
    lower_corner = root.find(".//gml:lowerCorner", ns)
    upper_corner = root.find(".//gml:upperCorner", ns)

    if lower_corner is not None and upper_corner is not None:
        lower_lat, lower_lon = lower_corner.text.split(" ")[:2]
        upper_lat, upper_lon = upper_corner.text.split(" ")[:2]

        lower_x, lower_y = to_cartesian_point(float(lower_lat), float(lower_lon), args.dsm_epsg)
        upper_x, upper_y = to_cartesian_point(float(upper_lat), float(upper_lon), args.dsm_epsg)

        if lower_x is None or upper_x is None or lower_y is None or upper_y is None:
            print("Coordinate transformation failed.")
            return

        bbox = (lower_x, lower_y, upper_x, upper_y)
        dsm_dir: str = os.path.expanduser(args.dsm_dir)
        matching_files = search_dsm_files(bbox, dsm_dir)

        if matching_files:
            print("Matching LAS files:")
            for file in matching_files:
                print(file)
        else:
            print("No matching LAS files found.")
    else:
        print("Corners not found in the CityGML file")


if __name__ == "__main__":
    main()
