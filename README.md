
# LOD2建築物モデル自動作成ツールv2.0

![image 8](https://user-images.githubusercontent.com/79615787/227534529-f858e8e7-1c56-49de-a5ab-5be177c1a0a9.png)


## ドキュメントを読む手順
1. [環境構築 & 実行方法](#環境構築実行方法)
2. [処理手順（要ドキュメント作成）](#処理手順)
3. [ドキュメントの限界](#ドキュメントの限界)、[コスト限界](#コスト限界)、[技術的限界](#技術的限界)
4. [なぜそこが技術的限界かに対する参考資料で深掘り](#2023年までの開発履歴の参照)


## 環境構築、実行方法

### [AWS EC2 Ubuntu 20.04 での環境構築のガイド](./docs/tutorials/setup-linux/README.md)
### [Docker Image のビルドとテスト方法](./docs/tutorials/docker/README.md)
### [VSCode Remote 設定ガイド](./docs/tutorials/vscode-remote/README.md)
### [便利な VSCode Extensions](./docs/tutorials/vscode-extensions/README.md)
### [開発ツールの利用ガイド](./docs/tutorials/dev-tools/README.md)


## 処理手順
| 順番 | 相対パス | 説明 | 内部処理の詳細 | 備考 |
|-|-|-|-|-|
| 1 | `./` | LOD2建築物自動作成ツール | [実行フロー](./docs/tutorials/tools/bldg-lod2-tool/README.md) | |
| - | ~~`./tools/SuperResolution/RoofSurface/`~~ | ~~屋根面視認性向上ツール~~ | | |
| 2 | `./tools/SuperResolution/WallSurface/` | 壁面視認性向上ツール | （要ドキュメント作成） | |
| 3 | `./tools/DeblurGANv2` | テクスチャ鮮明化ツール | [実行フロー](./docs/tutorials/tools/DeblurGANv2/README.md) | |
| 4 | `./tools/UnsharpMask/` | テクスチャシャープ化ツール | [実行フロー](./docs/tutorials/tools/UnsharpMask/README.md) | |
| 5 | `./tools/Real-ESRGAN/` | テクスチャ解像度向上ツール | [実行フロー](./docs/tutorials/tools/Real-ESRGAN/README.md) | |
| - | ~~`./tools/Atlas_Prot/`~~ | ~~テクスチャアトラス化ツール~~ | （要ドキュメント作成） | バグが多すぎて、使用中止 |


## [デバッグモードの使い方](./docs/tutorials/debug-mode/README.md)

## ドキュメントの限界(#2023年までの開発履歴の参照)）
### ${\text{\color{red}読む人にとってどこが重要な部分かのフォーカスがない。}}$
- 読む人が誰かを想定していない。
  - 辞書式で、すでに開発をしっている人にしか理解できない。
    - 初心者のためのドキュメントは、辞書式ではなく、前後の順序があるストーリ式がいい。
    - ドキュメント読む順番に注意。
  - 開発初心者を理解させるためではない。
  - 何かの成果発表。つまり、すでに知っている人に見せるためのもの。
- エントリポイントがない。
  - チュートリアルがない。
- ドキュメントの優先順位がない。
  - 読む順序がまるでわからない。重要度的になにが前が後かわからん。
  - なにが参考資料か、開発に必須かなどの重要度の表現がない。

## コスト限界
### ${\text{\color{red}品質の悪いDSMファイル}}$
- コストの事情で、品質の悪いDSMファイルを使って 3D モデルを生成ことになっている
- 一部の地域のDSMファイルは、点群のz座標が陥没していて、地面の高さになっている
  - 品質のいい地域は建物の 1% ~ 5% 以上陥没されている
  - 品質の悪い地域は建物の 10% ~ 50% 以上陥没されていて、使えないレベル
  - いい地域
- DSM の点群の座標は 0.25m 間隔で測量されたもので、誤差範囲もそれに比例する

## 技術的限界
### ${\text{\color{red}AIや数学的な処理の説明不足}}$
- コードが読めない（俗人化されている）
  - 丁寧な説明があっても理解できるかどうか
  - それを理由で、AIや数学的な処理ではない他のコードも可読性がすごく悪い
- 処理の中間過程がわからない
  - 評価の自動化が難しい
  - どこで問題が発生したかわからない
  - 再利用が難しい
- 最適化されている部分は未知のバグが多いから注意（コード汚すぎて読めないから指摘が入りにくい。）

### ${\text{\color{red}コアレベルが問題だらけ ＋ 問題だらけのコアへの依存性がすごく高い}}$
- AIや数学的処理は、とりあえず微調整が効かない
  - それを補うために、パッチにパッチを繰り返してコードがすごく汚い
- AIや数学的処理を変更する場合、その下の開発はほぼ全てやり直しになる確率が高い
  - 最適化されたコードは依存性が高くて、再利用が難しい
- 連鎖的に最適化されていて「そこだけ修正」は、基本的に不可能。
- なんで「そこだけ修正」ができないかの納得行く説明をするために、コミュニケーションコストがものすごく高い。
  - コアのバグから始まるなのパッチの連鎖を全て理解して説明できるようにすること
  - パッチそれぞれに対して、説明のために再現条件などを含めて証拠を作ること
  - 最適化レベルのパッチを、一般人レベルで説明するための資料を作ること

### ${\text{\color{red}問題のコア1 : HEATの屋根線追出AI}}$
- ２重屋根の建築物はほとんどの場合、屋根線追出が不完全
- 隣のビルや高い木の影で、屋根の色が変わると、変わった色の部分を違う屋根だと認識し、屋根線がおかしくなる
- 一部の地域の屋根画像を持って学習したモデルであって、全国対象で通じるかはテストしていない

### ${\text{\color{red}問題のコア2 : 3Dモデル生成部}}$
- 「一般ビルの建物の3Dモデル生成」と「複雑屋根の建物の3Dモデル生成」でコアが分離されている
  - HEATの不完全な屋根線をパッチをするため
- 「複雑屋根の建物の3Dモデル生成」のコア
  - ２重屋根の建築物に対して壁を生成する処理ができてないため、再設計が必要
    - 「そこだけ修正」は不可能

## 用再開発
### ${\text{\color{red}テクスチャアトラス化ツール}}$
- 一部のテキストファイルが処理されていない
- obj ファイルがある場合、obj ファイルの中にあるテクスチャ座標の変更プロセスがない
- コードが汚すぎて分析できない

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
