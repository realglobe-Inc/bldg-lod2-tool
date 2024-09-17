import os
from typing import Union
import lxml
import shutil
import glob
import pickle

import numpy as np
from numpy.typing import NDArray
from pyproj import Transformer

from ..thirdparty import plateaupy as plapy
from .coordinateconverter import CoordinateConverter
from .coordinateconverter import CoordinateConverterException
from .objinfo import BldElementType, ObjInfo
from .parammanager import ParamManager
from .config import Config
from .log import Log, ModuleType, LogLevel
from .resulttype import ProcessResult, ResultType


class CityGmlManager:
  """CityGML建物情報クラス
  """

  class BuildInfo:
    """CityGML出力建物形状情報クラス(建物毎)
    """
    class Lod2Info:
      """CityGML建物形状座標クラス(ポリゴン毎)
      """

      def __init__(self):
        """コンストラクタ
        """
        self.face_type = ''      # ポリゴン種別
        self.id_base = ''        # idベース
        self.poslist = []        # 建物形状座標リスト
        self.tex_coordlist = []  # テクスチャ画像座標リスト

    def __init__(self):
      """コンストラクタ
      """
      self.build_id = ''     # 建物ID(OBJファイル名は同一)
      self.lod0_poslist = []
      self.lod2_info = []
      self.tex_img_uri = None
      self.create_result = ResultType.SUCCESS         # 最終結果
      self.read_lod0_model = ProcessResult.ERROR      # LOD0モデルの読み込み結果
      self.create_lod2_model = ProcessResult.ERROR    # LOD2モデルの作成結果
      self.double_point = ProcessResult.SKIP          # 連続頂点重複検査結果
      self.solid = ProcessResult.SKIP                 # ソリッド閉合検査結果
      self.non_plane = ProcessResult.SKIP             # 非平面検査結果
      self.zero_area = ProcessResult.SKIP             # 面積0ポリゴン検査結果
      self.intersection = ProcessResult.SKIP          # 自己交差/自己接触検査結果
      self.face_intersection = ProcessResult.SKIP     # 地物内面同士交差検査結果
      self.paste_texture = ProcessResult.SKIP         # テクスチャ貼付け結果
      # CityGMLファイル名(サマリーファイルの出力用(複数ファイル対応で追加))
      self.citygml_filename = ''

    def append_Lod2Info(self, face_type, poslist, coordlist, id):
      """ポリゴン情報追加

      Args:
          face_type (BldElementType): ポリゴン種別
          poslist (Point[]): 建物形状座標リスト
          coordlist (Point[]): テクスチャ画像リスト
          id (string): ポリゴンのIDベース
      """
      lod2info = self.Lod2Info()
      lod2info.face_type = face_type
      lod2info.poslist = poslist
      lod2info.tex_coordlist = coordlist
      lod2info.id_base = id
      self.lod2_info.append(lod2info)

  def __init__(self, param_manager: ParamManager) -> None:
    """コンストラクタ

    Args:
        param_manager (ParamManager): パラメータ情報
    """
    self.param_manager = param_manager      # パラメータ情報
    self.citygml_info: list[self.BuildInfo] = []
    self.lod1_file_path = ''
    self._obj_info = ObjInfo()
    self._nsmap = None
    self._app_ns = ''
    self._bldg_ns = ''
    self._gml_ns = ''

    self._face_type = {BldElementType.ROOF: ["RoofSurface", "roof_"],
                       BldElementType.WALL: ["WallSurface", "wall_"],
                       BldElementType.GROUND: ["GroundSurface", "ground_"]}

  def read_file(
      self,
      file_name: str,
      target_coord_areas: Union[list[list[list[float]]], None] = None,
      debug_mode: bool = False
  ) -> tuple[ResultType, Union[list[BuildInfo], None]]:
    """CityGMLファイル読み込み

    Args:
      file_name       (str): ファイル名(拡張子付き)
      target_coord_areas (list[list[list[float]]] | None): 建築物選択範囲
      debug_mode (bool, optional): デバッグモード (Default: False)

    Returns:
      BuildInfo[]: CityGML建物情報リスト
    """
    try:
      restype = ResultType.SUCCESS

      # 平面直角座標系から経緯度座標系に変換
      self._trans_coordsys = CoordinateConverter(self.param_manager.las_coordinate_system)

      input_file_path = os.path.join(self.param_manager.citygml_folder_path, file_name)
      if not os.path.isfile(input_file_path):
        # 入力CityGMLファイルなし
        raise FileNotFoundError

      self.lod1_file_path = input_file_path
      filename_without_ext = os.path.splitext(self.lod1_file_path)[0]
      city_gml_cache_file_path = f"{filename_without_ext}.pkl"

      # デバッグ : CityGMLファイル読み込みを早くするため、pickle でキャッシュ化
      # 建物検索範囲を変更する場合、キャッシュされているファイルの削除が必要
      if os.path.exists(city_gml_cache_file_path) and debug_mode is True:
        with open(city_gml_cache_file_path, "rb") as f:
          self.citygml_info = pickle.load(f)

        return restype, self.citygml_info

      plbld = plapy.plbldg(self.lod1_file_path)

      if not plbld.buildings:
        raise Exception('Get err lod1citygml')

      target_geo_areas = None
      if target_coord_areas is not None:
        target_geo_areas = []
        for target_coord_area in target_coord_areas:
          target_geo_area = self._get_target_geo_area(target_coord_area)
          target_geo_areas.append(target_geo_area)

      not_found_outline_num = 0
      for bldg in plbld.buildings:
        binfo = self.BuildInfo()
        # 建物ID
        binfo.build_id = bldg.id
        # citygmlファイル名
        binfo.citygml_filename = file_name

        # 建物外形
        if bldg.lod0RoofEdge:
          binfo.lod0_poslist = bldg.lod0RoofEdge[0]
          binfo.read_lod0_model = ProcessResult.SUCCESS

        elif bldg.lod0FootPrint:
          binfo.lod0_poslist = bldg.lod0FootPrint[0]
          binfo.read_lod0_model = ProcessResult.SUCCESS

        else:
          # 建物外形が見つからない場合
          binfo.read_lod0_model = ProcessResult.ERROR
          not_found_outline_num += 1
          Log.output_log_write(LogLevel.WARN, ModuleType.INPUT_CITYGML, f"Outline not found {bldg.id}")
          restype = ResultType.WARN

        if target_geo_areas is None:
          self.citygml_info.append(binfo)
        else:
          for target_geo_area in target_geo_areas:
            # 緯度（latitude）の最小値と最大値を取得
            target_geo_area_lat_min = np.min(target_geo_area[:, 0])
            target_geo_area_lat_max = np.max(target_geo_area[:, 0])

            # 経度（longitude）の最小値と最大値を取得
            target_geo_area_lon_min = np.min(target_geo_area[:, 1])
            target_geo_area_lon_max = np.max(target_geo_area[:, 1])

            # 緯度（latitude）の最小値と最大値を取得
            binfo_lat_min = np.min(binfo.lod0_poslist[:, 0])
            binfo_lat_max = np.max(binfo.lod0_poslist[:, 0])

            # 経度（longitude）の最小値と最大値を取得
            binfo_lon_min = np.min(binfo.lod0_poslist[:, 1])
            binfo_lon_max = np.max(binfo.lod0_poslist[:, 1])

            # 建物の領域と検索領域が被っているか
            if (target_geo_area_lat_max >= binfo_lat_min and
                target_geo_area_lat_min <= binfo_lat_max and
                target_geo_area_lon_max >= binfo_lon_min and
                target_geo_area_lon_min <= binfo_lon_max
                ):

              self.citygml_info.append(binfo)
              break

      if len(self.citygml_info) == not_found_outline_num:
        raise Exception('All outline not found')

      # デバッグ : CityGMLファイル読み込みを早くするため、pickle でキャッシュ化
      if debug_mode is True:
        with open(city_gml_cache_file_path, "wb") as f:
          pickle.dump(self.citygml_info, f)

      return restype, self.citygml_info

    except FileNotFoundError:
      Log.output_log_write(LogLevel.MODEL_ERROR, ModuleType.INPUT_CITYGML, 'File not found')
      return ResultType.ERROR, None

    except CoordinateConverterException:
      Log.output_log_write(LogLevel.MODEL_ERROR, ModuleType.INPUT_CITYGML, 'Las coordinate system error')
      return ResultType.ERROR, None

    except Exception as e:
      Log.output_log_write(LogLevel.MODEL_ERROR, ModuleType.INPUT_CITYGML, e)
      return ResultType.ERROR, None

  def write_file(self, file_name: str) -> ResultType:
    """CityGMLファイル書き出し

    Args:
        file_name (str): ファイル名(拡張子付き)

    Returns:
        ResultType: 処理結果

    Note:
        出力ファイルパス
        [self.param_manager.output_folder_path] / [file_name]
    """
    try:
      # 出力CityGMLファイルパス
      output_file_path = os.path.join(self.param_manager.output_folder_path, file_name)

      # OBJ→CityGMLへ情報コピー
      ret = self._copy_objdata(Config.OUTPUT_TEX_OBJDIR)

      # CityGMLファイル書き出し
      if ret != ResultType.ERROR:
        plobj = plapy.plobj()
        tree, root = plobj.loadFile(self.lod1_file_path)
        self._nsmap = plobj.removeNoneKeyFromDic(root.nsmap)
        self._app_ns = "{" + self._nsmap['app'] + "}"
        self._bldg_ns = "{" + self._nsmap['bldg'] + "}"
        self._gml_ns = "{" + self._nsmap['gml'] + "}"

        wr_idx = 0
        surf_ret = []

        blds = tree.xpath('/core:CityModel/core:cityObjectMember/bldg:Building', namespaces=self._nsmap)

        # テクスチャ情報がある場合
        tex_appmem_elem = lxml.etree.Element(f"{self._app_ns}appearanceMember")
        tex_app_elem = lxml.etree.SubElement(tex_appmem_elem, f"{self._app_ns}Appearance")
        tex_theme_elem = lxml.etree.SubElement(tex_app_elem, f"{self._app_ns}theme")
        tex_theme_elem.text = "rgbTexture"

        for bld in blds:
          # 建物IDが一致する建物形状情報クラスを探す
          target_cgml = None
          for citygml in self.citygml_info:
            if citygml.read_lod0_model is ProcessResult.ERROR:
              # LOD0データの取得に失敗しているものは除外
              continue

            if bld.get(f"{self._gml_ns}id")\
               == citygml.build_id:
              target_cgml = citygml
              break
          if target_cgml is None:
            continue

          # lod2Solidの追加位置検索(lod1Solidの次)
          for num, bld_elem, in enumerate(list(bld)):
            if "lod1Solid" in bld_elem.tag:
              wr_idx = num
              break

          # lod2Solidのエレメント作成と追加
          if len(target_cgml.lod2_info):
            lod2_elem = self._create_lod2solid_elem(
                target_cgml.lod2_info)
            wr_idx += 1
            bld.insert(wr_idx, lod2_elem)

            # bldg:boundedByのエレメント作成と追加
            for info in target_cgml.lod2_info:
              bounded_elem = self._create_boundedby_elem(info)
              wr_idx += 1
              bld.insert(wr_idx, bounded_elem)

          # app:surfaceDataMemberのエレメント作成
          if target_cgml.tex_img_uri is not None:
            surf_ret.append(
                self._create_surfacedata_elem(
                    tex_app_elem,
                    target_cgml.tex_img_uri,
                    target_cgml.lod2_info))

        # app:surfaceDataMemberエレメント追加
        if any(surf_ret):
          root.append(tex_appmem_elem)

        # LoD2 CityGML書き出し
        lxml.etree.indent(root, space="\t")  # tab区切り
        tree.write(output_file_path,
                   pretty_print=True,
                   xml_declaration=True,
                   encoding="utf-8")
      else:
        shutil.copy(
            self.lod1_file_path, self.param_manager.output_folder_path)

      # LoD1 テクスチャデータがある場合は出力先にコピー
      in_dir = self.param_manager.citygml_folder_path
      out_dir = self.param_manager.output_folder_path
      base_name = (os.path.splitext(file_name)[0]).split('_op')[0]

      # 入力ディレクトリ内からフォルダ検索
      in_files = os.listdir(in_dir)
      files_dir = [f for f in in_files
                   if os.path.isdir(os.path.join(in_dir, f))]
      # 該当フォルダの検索
      in_texdir = f"${base_name}_appearance"
      l_in = [s for s in files_dir if in_texdir == s]

      if len(l_in) == 1:
        # 出力後フォルダ名
        # [メッシュコード]_[地物型]_[CRS]_[オプション]_appearance
        out_texdir = os.path.join(out_dir, in_texdir)

        if not os.path.isdir(out_texdir):
          os.mkdir(out_texdir)

        for getfile in glob.glob(os.path.join(in_dir, l_in[0], "*")):
          filename = os.path.basename(getfile)
          shutil.copy(getfile, os.path.join(out_texdir, filename))

      return ret

    except Exception as e:
      shutil.copy(
          self.lod1_file_path, self.param_manager.output_folder_path)
      Log.output_log_write(LogLevel.MODEL_ERROR,
                           ModuleType.OUTPUT_CITYGML, e)
      return False

  def _copy_objdata(self, obj_dir):
    """OBJファイル情報をCityGML形式にコピー

    Args:
        obj_dir (string): OBJフォルダパス

    Returns:
        bool: コピー結果
    """
    try:
      if not os.path.isdir(obj_dir):
        # OBJフォルダなし
        raise FileNotFoundError
      num_noobj = 0
      for num, citygml in enumerate(self.citygml_info, 1):
        obj_path = os.path.join(obj_dir, f'{citygml.build_id}.obj')
        if os.path.exists(obj_path):
          self._obj_info.read_file(obj_path)
          # 建物形状座標・テクスチャ座標の取得
          self._get_objdata(citygml)
          # テクスチャ画像URIの取得
          mtl_info = self._obj_info.get_mtl_info()
          if mtl_info is not None and citygml.build_id in mtl_info:
            citygml.tex_img_uri = mtl_info[citygml.build_id].map_kd
        else:
          # OBJファイルが見つからない場合
          Log.output_log_write(LogLevel.WARN, ModuleType.OUTPUT_CITYGML, f"Objfile not found {obj_path}")
          num_noobj += 1

      if num == num_noobj:
        # 全てのOBJファイルが見つからない場合
        raise Exception('All objfile not found')
      else:
        return ResultType.SUCCESS

    except FileNotFoundError:
      Log.output_log_write(LogLevel.MODEL_ERROR, ModuleType.OUTPUT_CITYGML, 'Folder not found')
      return ResultType.ERROR

    except Exception as e:
      Log.output_log_write(LogLevel.MODEL_ERROR, ModuleType.OUTPUT_CITYGML, e)
      return ResultType.ERROR

  def _get_objdata(self, citygml):
    """ポリゴン種別ごとに建物形状座標・テクスチャ画像座標の取得を行う

    Args:
        citygml (BuildInfo[]): CityGML建物情報リスト
    """
    num = 0
    for facetype in self._face_type.keys():
      texture_list_w = self._obj_info.get_texture_list(facetype)
      polygon_list_w = self._obj_info.get_polygon_list(facetype)
      if polygon_list_w:
        for (poly, tex) in zip(polygon_list_w, texture_list_w):
          id = citygml.build_id + "_" + str(num)
          citygml.append_Lod2Info(facetype, poly, tex, id)
          num += 1

  def _create_lod2solid_elem(self, lod2_info):
    """lod2Solidエレメント(LOD2建物テクスチャのポリゴンID群)作成

    Args:
        lod2_info (Lod2Info[]): 建物形状座標オブジェクトリスト

    Returns:
        Element: 作成したlod2Solidエレメント
    """
    elem1 = lxml.etree.Element(self._bldg_ns + "lod2Solid")
    elem2 = lxml.etree.SubElement(elem1, f"{self._gml_ns}Solid")
    elem3 = lxml.etree.SubElement(elem2, f"{self._gml_ns}exterior")
    elem4 = lxml.etree.SubElement(elem3, f"{self._gml_ns}CompositeSurface")
    for info in lod2_info:
      lxml.etree.SubElement(
          elem4,
          f"{self._gml_ns}surfaceMember",
          {lxml.etree.QName(self._nsmap['xlink'], 'href'): f"#texture_{info.id_base}"})
    return elem1

  def _create_boundedby_elem(self, info):
    """boundedByエレメント(LOD2建物形状)作成

    Args:
        info (Lod2Info): 建物形状座標オブジェクト

    Returns:
        Element: 作成したboundedByエレメント
    """
    str_list = []
    elem1 = lxml.etree.Element(self._bldg_ns + "boundedBy")
    elem2 = lxml.etree.SubElement(
        elem1,
        self._bldg_ns + self._face_type[info.face_type][0],
        {f"{self._gml_ns}id": self._face_type[info.face_type][1] + info.id_base}
    )
    elem3 = lxml.etree.SubElement(elem2, f"{self._bldg_ns}lod2MultiSurface")
    elem4 = lxml.etree.SubElement(elem3, f"{self._gml_ns}MultiSurface")
    elem5 = lxml.etree.SubElement(elem4, f"{self._gml_ns}surfaceMember")
    elem6 = lxml.etree.SubElement(
        elem5,
        f"{self._gml_ns}Polygon",
        {f"{self._gml_ns}id": f"texture_{info.id_base}"}
    )
    elem7 = lxml.etree.SubElement(elem6, f"{self._gml_ns}exterior")
    elem8 = lxml.etree.SubElement(
        elem7,
        f"{self._gml_ns}LinearRing",
        {f"{self._gml_ns}id": f"shape_{info.id_base}"}
    )
    elem9 = lxml.etree.SubElement(elem8, f"{self._gml_ns}posList")

    for pos in info.poslist:
      lat, lon = self._trans_coordsys.to_polar(pos.x, pos.y)

      str_list.append(str(lat))
      str_list.append(str(lon))
      str_list.append(str(pos.z))

    # 先頭座標を末尾に追加する
    lat, lon = self._trans_coordsys.to_polar(info.poslist[0].x, info.poslist[0].y)
    str_list.append(str(lat))
    str_list.append(str(lon))
    str_list.append(str(info.poslist[0].z))

    elem9.text = ' '.join(str_list)

    return elem1

  def _create_surfacedata_elem(self, app_elem, uri, lod2_info):
    """surfaceDataMemberエレメント(テクスチャ情報)作成

    Args:
        app_elem (Element): Appearanceエレメント
        uri (string): テクスチャ画像URI
        lod2_info (Lod2Info[]): 建物形状座標オブジェクトリスト

    Returns:
        bool: テクスチャ座標あり(True)/テクスチャ座標なし(False)
    """
    elem5 = None
    elem1 = lxml.etree.SubElement(app_elem, f"{self._app_ns}surfaceDataMember")
    elem2 = lxml.etree.SubElement(elem1, f"{self._app_ns}ParameterizedTexture")
    elem3 = lxml.etree.SubElement(elem2, f"{self._app_ns}imageURI")
    elem3.text = uri  # URI
    elem4 = lxml.etree.SubElement(elem2, f"{self._app_ns}mimeType")
    elem4.text = "image/jpg"

    for info in lod2_info:
      str_list = []
      for coordlist in info.tex_coordlist:
        if coordlist is not None:
          str_list.append(str(coordlist.x))
          str_list.append(str(coordlist.y))
      # 先頭座標を末尾に追加する
      if info.tex_coordlist[0] is not None:
        str_list.append(str(info.tex_coordlist[0].x))
        str_list.append(str(info.tex_coordlist[0].y))

      if str_list:
        elem5 = lxml.etree.SubElement(
            elem2, f"{self._app_ns}target", {'uri': f"#texture_{info.id_base}"}
        )
        elem6 = lxml.etree.SubElement(elem5, f"{self._app_ns}TexCoordList")
        elem7 = lxml.etree.SubElement(
            elem6, f"{self._app_ns}textureCoordinates", {'ring': f"#shape_{info.id_base}"}
        )
        elem7.text = ' '.join(str_list)

    # 一つでも座標配列が見つかれば成功
    if elem5 is not None:
      return True
    else:
      return False

  def _get_target_geo_area(self, target_coord_area: list[list[float]]) -> NDArray[np.float_]:
    """建築物検索範囲から緯度経度範囲に変換

    Args:
        target_coord_area (list[list[float]]): 建築物検索範囲

    Returns:
        NDArray[np.float_]: 建築物の緯度経度範囲
    """
    target_geo_area = []
    for target_coord_spot in target_coord_area:
      pos_a, pos_b, epsg_code = target_coord_spot

      # 緯度経度座標の場合はそのまま
      if epsg_code == 6668:
        target_geo_area.append([pos_a, pos_b])

      # 平面直角座標 -> 緯度経度座標
      else:
        polar_trans = Transformer.from_crs(f'epsg:{epsg_code}', 'epsg:6668', always_xy=True)
        lon, lat = polar_trans.transform(pos_a, pos_b)
        target_geo_area.append([lat, lon])

    return np.array(target_geo_area, dtype=np.float_)
