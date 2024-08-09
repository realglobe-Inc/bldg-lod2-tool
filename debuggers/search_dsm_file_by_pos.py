import os
import argparse
from typing import Union

import laspy
import numpy as np


def is_bbox_overlap(bbox1: tuple[float], bbox2: tuple[float]):
    """
    Check if two bounding boxes overlap.
    bbox1 and bbox2: (min_x, min_y, max_x, max_y)
    """
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])


def get_las_bbox(las_file: str) -> Union[tuple[float], None]:
    """
    Get the bounding box (min_x, min_y, max_x, max_y) of the LAS file.
    """
    try:
        with laspy.open(las_file) as f:
            header = f.header
            min_x, max_x = header.mins[0], header.maxs[0]
            min_y, max_y = header.mins[1], header.maxs[1]
            return (min_x, min_y, max_x, max_y)
    except Exception as e:
        print(f"Error reading {las_file}: {e}")
        return None


def search_dsm_files(bbox: tuple[float], dsm_dir: str):
    """
    Search for LAS files that overlap with the given bounding box.
    """
    matching_files: list[str] = []

    for root, _dirs, files in os.walk(dsm_dir):
        for file in files:
            if file.endswith(".las"):
                file_path = os.path.join(root, file)
                las_bbox = get_las_bbox(file_path)
                if las_bbox is None:
                    continue

                if las_bbox and is_bbox_overlap(las_bbox, bbox):
                    matching_files.append(file_path)

    return matching_files


def main():
    parser = argparse.ArgumentParser(description="Search LAS files by bounding box.")
    parser.add_argument("min_x", type=float, help="Minimum X coordinate")
    parser.add_argument("min_y", type=float, help="Minimum Y coordinate")
    parser.add_argument("max_x", type=float, help="Maximum X coordinate")
    parser.add_argument("max_y", type=float, help="Maximum Y coordinate")
    parser.add_argument("dsm_dir", type=str, help="Directory containing DSM (LAS) files")

    args = parser.parse_args()
    # python3 search_dsm_file_by_pos.py -21146 -34454 -21132 -34440 ~/DSM/DSM/

    bbox: tuple[float] = (args.min_x, args.min_y, args.max_x, args.max_y)
    dsm_dir: str = os.path.expanduser(args.dsm_dir)
    matching_files = search_dsm_files(bbox, dsm_dir)

    if matching_files:
        print("Matching LAS files:")
        for file in matching_files:
            print(file)
    else:
        print("No matching LAS files found.")

if __name__ == "__main__":
    main()
