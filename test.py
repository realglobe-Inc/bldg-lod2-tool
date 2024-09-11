import numpy as np
from scipy.spatial import KDTree

points = np.array([
    [37.454011884736246, 3.142918568673425, 19.26094938462863],
    [95.07143064099162, 63.64104112637804, 2.524198949851465],
    [73.1993941811405, 31.435598107632668, 4.848861422838413],
    [59.86584841970366, 50.85706911647028, 26.956625655812378],
    [15.601864044243651, 90.7566473926093, 18.1928717897877],
    [15.599452033620265, 24.929222914887493, 0.27591154849888944],
    [5.8083612168199465, 41.038292303562976, 3.044146285980963],
    [86.61761457749351, 75.55511385430486, 19.905053073241675],
    [60.11150117432088, 22.879816549162246, 0.1518475153865606],
    [70.80725777960456, 7.697990982879299, 4.82424154252496],
    [2.0584494295802447, 28.9751452913768, 16.462013680997583],
    [96.99098521619943, 16.122128725400444, 20.756855930780798],
    [83.24426408004217, 92.96976523425731, 19.558837785078016],
    [21.233911067827616, 80.8120379564417, 6.728079283816793],
    [18.182496720710063, 63.34037565104234, 21.365376640426074],
    [18.34045098534338, 87.14605901877177, 7.117472624904002],
    [30.42422429595377, 80.36720768991145, 9.76199094477803],
    [52.475643163223786, 18.657005888603585, 22.394742153540726],
    [43.194501864211574, 89.25589984899777, 19.48898697141644],
    [29.122914019804192, 53.93422419156507, 25.47670231482534],
    [61.18528947223795, 80.74401551640625, 19.7283867690103],
    [13.949386065204184, 89.60912999234932, 17.049258100064147],
    [29.214464853521815, 31.800347497186387, 2.8102430348427743],
    [36.63618432936917, 11.005192452767677, 11.031474091783005],
    [45.606998421703594, 22.793516254194166, 7.9560710304517634],
    [78.51759613930136, 42.71077886262563, 7.319689301372508],
    [19.967378215835975, 81.80147659224932, 29.190316642573368],
    [51.42344384136116, 86.07305832563435, 11.792931740002812],
    [59.24145688620425, 0.6952130531190703, 26.761396655313398],
    [4.645041271999773, 51.07473025775657, 18.934158779917887],
    [60.75448519014384, 41.7411003148779, 23.844339106249453],
    [17.052412368729154, 22.210781047073024, 15.079112793155764],
    [6.505159298527952, 11.98653673336828, 17.307116538790773],
    [94.88855372533332, 33.7615171403628, 14.775530814565917],
    [96.56320330745594, 94.29097039125192, 5.857289633941335],
    [80.83973481164611, 32.320293202075526, 21.67356345784516],
    [30.46137691733707, 51.87906217433661, 8.423170873225674],
    [9.767211400638388, 70.30189588951778, 0.7294789929436152],
    [68.42330265121569, 36.3629602379294, 19.364168877215036],
    [44.01524937396013, 97.17820827209607, 5.313320382211469],
    [12.203823484477883, 96.24472949421113, 28.21375753058743],
    [49.51769101112702, 25.178229582536417, 28.61785731007762],
    [3.4388521115218396, 49.72485058923854, 27.445931706613457],
    [90.9320402078782, 30.087830981676966, 11.104761007663331],
    [25.87799816000169, 28.484049437746762, 0.46369849586602285],
    [66.2522284353982, 3.6886947354532795, 27.849556877631763],
    [31.171107608941096, 60.956433397989684, 12.84552444951943],
    [52.00680211778108, 50.26790232288615, 28.999644571310085],
    [54.67102793432797, 5.147875124998935, 28.908599312677584],
    [18.485445552552704, 27.864646423661142, 25.590283664020802],
    [96.95846277645586, 90.82658859666537, 8.833466762087571],
    [77.51328233611146, 23.95618906669724, 11.552931858057757],
    [93.9498941564189, 14.48948720912231, 25.534100145505708],
    [89.48273504276489, 48.9452760277563, 9.50766015468833],
    [59.78999788110851, 98.56504541106007, 5.084782400582775],
    [92.18742350231169, 24.20552715115004, 16.704037873750504],
    [8.84925020519195, 67.21355474058785, 28.08464322482343],
    [19.59828624191452, 76.16196153287176, 20.88089390024919],
    [4.522728891053807, 23.763754399239968, 17.101835102680948],
    [32.53303307632643, 72.82163486118596, 2.9152948131230563],
    [38.8677289689482, 36.77831327192532, 18.45021680097509],
    [27.134903177389592, 63.23058305935795, 29.701615503127897],
    [82.87375091519293, 63.35297107608947, 4.202520457095721],
    [35.67533266935893, 53.57746840747585, 15.549889570912102],
    [28.093450968738075, 9.02897700544083, 26.321192157838663],
    [54.26960831582485, 83.5302495589238, 22.223058532626133],
    [14.092422497476264, 32.07800649717358, 20.91047222985804],
    [80.21969807540397, 18.651851039985424, 21.07452251961328],
    [7.455064367977082, 4.077514155476392, 10.784734536592655],
    [98.68869366005173, 59.08929431882418, 8.807755327934801],
    [77.22447692966574, 67.75643618422824, 24.28083466435541],
    [19.87156815341724, 1.6587828927856152, 24.30340184037542],
    [0.5522117123602399, 51.2093058299281, 26.012169557403112],
    [81.54614284548342, 22.649577519793795, 27.397216576694138],
    [70.68573438476172, 64.51727904094498, 15.340271965828133],
    [72.90071680409874, 17.436642900499145, 15.045488840615988],
    [77.12703466859458, 69.09377381024659, 23.948855369003255],
    [7.4044651734090365, 38.67353463005374, 19.498917923332954],
    [35.84657285442726, 93.67299887367345, 21.0590063177311],
    [11.586905952512971, 13.752094414599325, 23.87378008308303],
    [86.31034258755935, 34.10663510502585, 26.70016025452699],
    [62.329812682755794, 11.347352124058908, 10.139854705546075],
    [33.08980248526492, 92.46936182785628, 11.26748857919832],
    [6.355835028602364, 87.7339353380981, 2.81945819522607],
    [31.09823217156622, 25.79416277151556, 17.34840422988522],
    [32.518332202674706, 65.99840460341791, 1.0782682139022626],
    [72.96061783380641, 81.72222002012158, 13.967940543973805],
    [63.75574713552131, 55.52008115994623, 16.2793390412273],
    [88.72127425763266, 52.965057835600646, 8.596237563848533],
    [47.22149251619493, 24.18522909004517, 17.724997817070324],
    [11.959424593830171, 9.310276780589922, 0.9150074981714829],
    [71.3244787222995, 89.72157579533267, 1.1204456624764325],
    [76.07850486168974, 90.04180571633304, 24.678016819789747],
    [56.127719756949624, 63.31014572732679, 10.805719242337886],
    [77.0967179954561, 33.90297910487007, 3.8118153795565437],
    [49.379559636439076, 34.92095746126609, 15.667297801644132],
    [52.27328293819941, 72.59556788702393, 23.099806592958323],
    [42.75410183585496, 89.71102599525771, 6.4746308249052955],
    [2.541912674409519, 88.70864242651173, 18.686714274570008],
    [10.789142699330444, 77.98755458576238, 2.56042394981304],
], dtype=np.float_)


# 高さのしきい値
height_threshold = 0.5

# 壁を検出する関数


def identify_wall_points(points, height_threshold=0.5):
  wall_points = []
  tree = KDTree(points[:, :2])  # x, y 座標で隣接点を探す

  for i, point in enumerate(points):
    indices = tree.query_ball_point(point[:2], r=1.0)

    for j in indices:
      if i != j:
        height_diff = abs(point[2] - points[j][2])

        if height_diff >= height_threshold:
          if point[2] > points[j][2]:
            wall_points.append(point)
          else:
            wall_points.append(points[j])
          break

  return np.array(wall_points)

# 再帰的に囲まれた点を取得する関数 (複数のsub_treeに対応)


def extract_enclosed_points(points, wall_points, level=1, max_levels=10):
  if level > max_levels or len(points) == 0:
    return []

  enclosed_clusters = detect_enclosed_areas(points, wall_points)  # 複数の囲まれた領域を取得

  sub_trees = []

  for cluster in enclosed_clusters:
    # クラスター内の点に対して再帰的に処理
    remaining_points = points_excluding_cluster(points, cluster)  # クラスター外の点を次の処理に渡す
    sub_tree = extract_enclosed_points(remaining_points, wall_points, level + 1, max_levels)

    sub_trees.append({
        'level': level,
        'enclosed_cluster': cluster,
        'sub_tree': sub_tree  # 次の階層のサブツリー
    })

  return sub_trees

# 仮に "領域を囲む壁を見つける" 検出関数


def detect_enclosed_areas(points, wall_points):
  # ここに複数の囲まれた領域を検出するロジックを実装
  # 仮にランダムにいくつかの領域を返す (ダミーの例)
  num_clusters = 3  # 例として3つの囲まれた領域を検出したと仮定
  clusters = [points[np.random.choice(points.shape[0], size=10, replace=False)] for _ in range(num_clusters)]

  return clusters

# クラスター外の点を取得する関数


def points_excluding_cluster(points, cluster):
  return np.array([p for p in points if not any(np.array_equal(p, c) for c in cluster)])


# 実行
wall_points = identify_wall_points(points, height_threshold)
# tree_structure = extract_enclosed_points(points, wall_points)

# ツリー構造を出力
import pprint
pprint.pprint(wall_points)
