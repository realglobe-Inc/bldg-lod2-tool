
# LOD2建築物モデル自動作成ツールv2.0

![image 8](https://user-images.githubusercontent.com/79615787/227534529-f858e8e7-1c56-49de-a5ab-5be177c1a0a9.png)

## 環境構築 & 実行方法

### [AWS EC2 Ubuntu 20.04 での環境構築のガイド](./docs/tutorials/setup-linux/README.md)
### [VSCode Remote 設定ガイド](./docs/tutorials/vscode-remote/README.md)

## [デバッグモードの使い方](./docs/tutorials/debug-mode//README.md)

## 利用技術

本ツールは、Python(バージョン3.9)のインストールが必要です。

| 種別 | ライブラリ名 | ライセンス | 説明 |
| - | - | - | - |
| ライブラリ |alphashape|MIT License|点群外形形状作成ライブラリ|
|  |anytree|Apache 2.0|木構造ライブラリ|
|  |autopep8|MIT License|コーディング規約(PEP)準拠にソースコードを自動修正するフォーマッターライブラリ|
|  |coverage|Apache 2.0|カバレッジ取得ライブラリ|
|  |einops|MIT License|数値計算ライブラリ|
|  |flake8|MIT License|静的解析ライブラリ|
|  |jakteristics|BSD License|点群の幾何学的特徴量計算ライブラリ|
|  |laspy|BSD 2-Clause License|LASファイル処理ライブラリ|
|  |lxml|BSD 3-Clause License|xml処理ライブラリ|
|  |matplotlib|Python Software Foundation License|グラフ描画ライブラリ|
|  |MLCollections|Apache 2.0|機械学習ライブラリ|
|  |MultiScaleDeformableAttention|Apache 2.0|物体検出ライブラリ|
|  |NumPy|BSD 3-Clause License|数値計算ライブラリ|
|  |Open3D|MIT License|点群処理ライブラリ|
|  |opencv-python|MIT License|画像処理ライブラリ|
|  |opencv-contrib-python|MIT License|画像処理ライブラリ|
|  |Pytorch|BSD 3-Clause License|機械学習ライブラリ|
|  |plateaupy|MIT License|CityGML読み込みライブラリ|
|  |PyMaxflow|GNU General Public License version 3.0|GraphCut処理ライブラリ|
|  |pyproj|MIT License|地理座標系変換ライブラリ|
|  |PuLP|BSD License|数理最適化ライブラリ|
|  |scikit-learn|BSD 3-Clause License|機械学習ライブラリ|
|  |scipy|BSD 3-Clause License|統計や線形代数、信号・画像処理などのライブラリ|
|  |Shapely|BSD 3-Clause License|図形処理ライブラリ|
|  |Torch|BSD 3-Clause Lisence|機械学習ライブラリ|
|  |Torchvision|BSD 3-Clause Lisence|機械学習ライブラリ|

## フォルダ構成
| 相対パス |　詳細 |
|-|-|
| `./` | LOD2建築物自動作成ツール |
| `./tools/Atlas_Prot/` | 建物テクスチャアトラス化ツール |
| `./tools/SuperResolution/` | 建物テクスチャ視認性向上ツール |
| `./tools/SuperResolution/RoofSurface/` | 屋根面視認性向上ツール |
| `./tools/SuperResolution/WallSurface/` | 壁面視認性向上ツール |
| `./tools/SuperResolution/WallSurface/` | 壁面視認性向上ツール |
| `./tools/DeblurGANv2` | 画質向上ツール |
| `./tools/UnsharpMask/` | 画質のエッジシャープ化ツール |
| `./tools/Real-ESRGAN/` | 解像度向上ツール |

## ライセンス
- 本プロジェクトは[Auto-Create-bldg-lod2-tool](https://github.com/Project-PLATEAU/Auto-Create-bldg-lod2-tool/tree/f68a85bac55ff61d3c5c6192121513e7b7f77861)をフォークしたものです。
- 本ツールはGNU General Public License v3.0を適用します。
- 本ドキュメントは[Project PLATEAUのサイトポリシー](https://www.mlit.go.jp/plateau/site-policy/)（CCBY4.0および政府標準利用規約2.0）に従い提供されています。

## 注意事項
- 本レポジトリは参考資料として提供しているものです。動作保証は行っておりません。
- 予告なく変更・削除する可能性があります。
- 本レポジトリの利用により生じた損失及び損害等について、Realglobe はいかなる責任も負わないものとします。

## 2023年までの開発履歴の参照
- 2022年開発 | AI等を活用したLOD2自動生成ツールの開発及びOSS化技術検証レポート
https://www.mlit.go.jp/plateau/file/libraries/doc/plateau_tech_doc_0056_ver01.pdf
- 2023年開発 | AI等を活用したLOD2自動生成ツールの開発及びOSS化技術検証レポート
https://www.mlit.go.jp/plateau/file/libraries/doc/plateau_tech_doc_0061_ver01.pdf
- 2023年開発 | 3D都市モデルのテクスチャ高解像度化手法及び描画パフォーマンス向上に関する技術調査レポート
https://www.mlit.go.jp/plateau/file/libraries/doc/plateau_tech_doc_0062_ver01.pdf
- 2023年開発追記漏れがあり(要追記)
  - 屋根線取得部分で設計変更。DSMから屋根イメージを取得するため、オルソ画像の入力はなしになった

## 2024年以後の開発履歴の参照

## 著作権

本プロジェクトの元のコードは以下の著作権に従います:

- 著作権 (C) 2024 国土交通省

このプロジェクトはGNU General Public License v3.0の下でライセンスされています。詳細は[LICENSEファイル](LICENSE)を参照してください。

### フォークに関する著作権

本プロジェクトは[Auto-Create-bldg-lod2-tool](https://github.com/Project-PLATEAU/Auto-Create-bldg-lod2-tool/tree/f68a85bac55ff61d3c5c6192121513e7b7f77861)をフォークしたものです。

新しい変更部分に関する著作権:

- 著作権 (C) 2024 Realglobe

本プロジェクトの新しい部分も、GPLv3の条件の下でライセンスされています。
