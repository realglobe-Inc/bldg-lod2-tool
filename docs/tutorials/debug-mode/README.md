# デバッグモードの使い方

## [設定パラメーター](../setup-linux/README.md#LOD2建築物モデル自動作成パラメーター修正)
- DebugMode を `true` に設定
- TargetCoordAreas に検索範囲を入力して処理対象を制限
  - 緯度経度座標 epsg_code : 6668
  - 平面直角座標 epsg_code : 6677(地域によって違う)
    - 地域コード : https://www.sinfonica.or.jp/faq/gis/minf_hzahyo.html
    - [地域コードから epsg_code 変換](../../../src/util/coordinateconverter.py)
- TargetBuildingIds で処理対象を制限

## デバッグモードの使用時、注意点
- `TargetCoordAreas` を変更した場合、キャッシュ削除が必要
  - [建物形状分類のキャッシュ](../../../src/createmodel/data)フォルダーの `building_class_cache.json` を削除
  - DSMキャッシュ削除パラメーター `DsmFolderPath` の下にある `pkl` ファイルを削除
  - CityGMLキャッシュ削除パラメーター `CityGMLFolderPath` の下にある `pkl` ファイルを削除
