

import copy
import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from collections import deque

last_merged_polygons = [[(2.0, 5.0), (9.0, 5.0), (10.0, 6.0), (12.0, 6.0), (15.0, 4.0), (16.0, 4.0), (19.0, 4.0), (19.0, 3.0), (17.0, 3.0), (16.0, 2.0), (16.0, 0.0), (1.0, 0.0), (1.0, 7.0)], [(16.0, 36.0), (15.0, 35.0), (1.0, 35.0), (0.0, 34.0), (0.0, 36.0)]]
current_merged_polygons = [[(9.0, 5.0), (10.0, 6.0), (12.0, 6.0), (15.0, 4.0), (16.0, 4.0), (19.0, 4.0), (19.0, 3.0), (17.0, 3.0), (16.0, 2.0), (16.0, 0.0), (1.0, 0.0), (1.0, 7.0), (2.0, 5.0)], [(16.0, 36.0), (15.0, 35.0), (1.0, 35.0), (0.0, 34.0), (0.0, 36.0)]]

last_merged_polygons_union = unary_union([Polygon(p) for p in last_merged_polygons])
current_merged_polygons_union = unary_union([Polygon(p) for p in current_merged_polygons])
difference = current_merged_polygons_union.difference(last_merged_polygons_union)
current_polygons = []
if isinstance(difference, MultiPolygon):
  for poly in difference:
    polygon = [coord for coord in poly.exterior.coords[:-1]]
    current_polygons.append(polygon)
elif isinstance(difference, Polygon):
  polygon = [coord for coord in difference.exterior.coords[:-1]]
  current_polygons.append(polygon)

print('start')
print(last_merged_polygons)
print(current_merged_polygons)
print(current_polygons)
