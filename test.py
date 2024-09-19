import open3d as o3d
import numpy as np
import random

# パラメータ設定
height_threshold = 0.5  # 隣接点間の高さ差の閾値
min_points_per_cluster = 10  # クラスタの最小点数
eps = 1.0  # クラスタリングの近傍距離

# 1. 点群データの読み込み
# 例としてランダムに生成した点群
np.random.seed(42)
points = np.random.rand(1000, 3)  # ランダムに1000点生成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 2. k近傍法を使用して各点の隣接点を取得
kdtree = o3d.geometry.KDTreeFlann(pcd)

# 3. 壁点のリスト
wall_points = []
non_wall_points = []

# 4. 各点について、隣接点との高さ差を確認して壁点を検出
for i in range(len(pcd.points)):
  [k, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], 6)  # 隣接点を6個取得
  neighbors = np.asarray(pcd.points)[idx[1:], :]  # 自分以外の隣接点を取得

  # 高さ差を計算 (z 座標の差)
  height_diffs = np.abs(neighbors[:, 2] - pcd.points[i][2])

  # 高さ差が閾値を超える点があれば、その点を壁点としてマーク
  if np.any(height_diffs > height_threshold):
    wall_points.append(pcd.points[i])
  else:
    non_wall_points.append(pcd.points[i])

# 5. 壁点をPointCloudオブジェクトに変換
wall_pcd = o3d.geometry.PointCloud()
wall_pcd.points = o3d.utility.Vector3dVector(np.array(wall_points))

# 6. DBSCAN クラスタリングを使って、壁点で囲まれた領域をクラスタリング
labels = np.array(wall_pcd.cluster_dbscan(eps=eps, min_points=min_points_per_cluster, print_progress=True))

# クラスタごとにランダムな色を生成
max_label = labels.max()

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
random_colors = np.random.rand(max_label + 1, 3)

# 各クラスタにランダムな色を割り当てる
colors = [random_colors[label] if label >= 0 else [0, 0, 0] for label in labels]
wall_pcd.colors = o3d.utility.Vector3dVector(colors)

# 7. 検出された壁点とクラスタの可視化
o3d.visualization.draw_geometries([wall_pcd], window_name="Wall Line Detection with Clusters",
                                  point_show_normal=False, width=800, height=600)
