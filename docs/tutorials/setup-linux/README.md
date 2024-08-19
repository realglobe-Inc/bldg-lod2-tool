# AWS EC2 Ubuntu 20.04 での環境構築のガイド

## 動作環境
| 項目               | 最小動作環境               | 推奨動作環境                   |
| ------------------ | ------------------------- | ------------------------------ |
| OS                 | Microsoft Windows 10 または 11 または Linux | 同左 |
| CPU                | Intel Core i5以上 | Intel Core i7以上 |
| メモリ             | 8GB以上 | 16GB以上 |
| GPU                | NVIDIA Quadro P620以上 | NVIDIA RTX 2080以上 |
| GPU メモリ         | 2GB以上 | 8GB以上 |

## インスタンス構築
- AWS cuda 11.3 がインストールされている AMIを選択して、EC2作成
  - 22.04 以上からは cuda 11.3 がインストールされないから注意
- インスタンススペック
  - g4dn.xlarge
  - SSD 500GB

## Python 3.9.19 インストール
```
git clone git://github.com/yyuu/pyenv.git ~/.pyenv

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

pyenv install 3.9.19
pyenv global 3.9.19
```

## プロジェクト clone
```
git clone --recurse-submodules https://github.com/realglobe-Inc/bldg-lod2-tool
```

## python 仮想環境の設定方法
- 以下を ~/.bashrc に追加
```
alias create_env='python -m venv $(basename $PWD)'
alias activate='source "$PWD/$(basename $PWD)/bin/activate"'
```

- 仮想環境の作成
```
# requirements.txt のあるフォルダーに移動して実行
# ./                       : LOD2建築物モデル自動作成ツール
# ./tools/SuperResolution  : 屋根面視認性向上ツールと壁面視認性向上ツール
# ./tools/DeblurGANv2      : 画質向上ツール
# ./tools/UnsharpMask      : 画質のエッジシャープ化ツール
# ./tools/Real-ESRGAN      : 解像度向上ツール
# ./tools/Atlas_Prot       : アトラス化ツール
create_env
```

- 仮想環境の開始
```
activate # requirements.txt のあるフォルダーに移動して実行
```

- 仮想環境の終了
```
deactivate # requirements.txt のあるフォルダーに移動して実行
```

## LOD2建築物モデル自動作成ツール

### プロジェクト内相対パスへ移動 : ./

### 依存ライブラリのインストール
```
pip install –r requirements.txt # 仮想環境の開始後
```

### 屋根全取得のモデルの拡張モジュールのインストール
```
python3 bldg-lod2-tool/src/createmodel/housemodeling/roof_edge_detection_model/thirdparty/heat/models/ops/setup.py build
python3 bldg-lod2-tool/src/createmodel/housemodeling/roof_edge_detection_model/thirdparty/heat/models/ops/setup.py install
```

### 建物分類用モデルの学習済みパラメーターをダウンロード
```
wget -O src/createmodel/data/classifier_parameter.pkl https://drive.google.com/file/d/1hs-DT4Y0ZtjdV9kJ438lvAPpJcfz_dE_/view?usp=drive_link
```

### 屋根線検出用モデルの学習済みパラメーターをダウンロード
```
wget -O src/createmodel/data/roof_edge_detection_parameter.pth https://drive.google.com/file/d/1QqxfS05a4T1_IdrzYle3iuBXjuyqFz-u/view?usp=drive_link
```

### バルコニー検出用モデルの学習済みパラメーターをダウンロード
```
wget -O src/createmodel/data/balcony_segmentation_parameter.pkl https://drive.google.com/file/d/1MINHffIvcooDOrQq3E4mBvdsgWUfzIi5/view?usp=drive_link
```

### LOD2建築物モデル自動作成のテスト用の入力データーのダウンロード
- [ブラウザからダウンロード(ファイル大きいから wget 不可)](https://drive.google.com/file/d/1UnxBL2MrDZaQ5EF44TXCFc-ZnVbp9Gf6/view)
- scp でインスタンスに転送
```
scp ~/Download/Auto-Create-bldg-lod2-tool-tutorial.zip ubuntu@xxx.xxx.xxx.xxx:~/
```
- インスタンス内部で圧縮解除
```
mkdir -p ~/Auto-Create-bldg-lod2-tool-tutorial
unzip ~/Auto-Create-bldg-lod2-tool-tutorial.zip -d ~/Auto-Create-bldg-lod2-tool-tutorial
```

### LOD2建築物モデル自動作成パラメーター修正
~/AutoCreateLod2_tutorial/LOD2Creator_tutorial/param.json
```
{
  "LasCoordinateSystem": 9,
  "DsmFolderPath": "~/AutoCreateLod2_tutorial/LOD2Creator_tutorial/dataset/DSM",
  "LasSwapXY": false,
  "CityGMLFolderPath": "~/AutoCreateLod2_tutorial/LOD2Creator_tutorial/dataset/CityGML",
  "TextureFolderPath": "~/AutoCreateLod2_tutorial/LOD2Creator_tutorial/dataset/RawImage",
  "RotateMatrixMode": 0,
  "ExternalCalibElementPath": "~/AutoCreateLod2_tutorial/LOD2Creator_tutorial/dataset/ExCalib/ExCalib.txt",
  "CameraInfoPath": "~/AutoCreateLod2_tutorial/LOD2Creator_tutorial/dataset/CamInfo/CamInfo.txt",
  "OutputFolderPath": "~/AutoCreateLod2_tutorial/output",
  "OutputOBJ": false,
  "OutputTexture": true,
  "OutputLogFolderPath": "~/AutoCreateLod2_tutorial/output",
  "DebugLogOutput": false,
  "PhaseConsistency": {
    "DeleteErrorObject": true,
    "NonPlaneThickness": 0.05,
    "NonPlaneAngle": 15
  },
  "TargetCoordAreas" : [
    [[35, 139, 6668], [36, 140, 6668]],
    [[35, 139, 6668], [-12516, -53774, 6677]],
    [[-12516, -53774, 6677], [-12516, -53774, 6677]]
  ]
}
```

### LOD2建築物モデル自動作成開始
```
python3 AutoCreateLod2.py ~/AutoCreateLod2_tutorial/LOD2Creator_tutorial/param.json
```


#### 必須パラメーター
| No |	キー名 |	値形式 | 説明 |
| -- | -- | -- | -- | 
|1|	LasCoordinateSystem| 数値	|航空写真DSM点群の平面直角座標系の番号です。1～19の数値にて指定します。未記入および1～19以外の値が入力された場合は無効とし、エラーメッセージを表示し、処理を中止します。
|2|	DsmFolderPath| 文字列 |航空写真DSM点群のフォルダパスを指定します。指定されたフォルダパスが存在しない場合は無効とし、エラーメッセージを表示し、処理を中止します。
|3|	LasSwapXY | 真偽値 |	LASファイルのXY座標を入れ替えて使用するか否かを切り替えるフラグです。設定値がtrueの場合は、LASファイルのXY座標を入れ替えます。システム内部座標系が、xが東方向、yが北方向の値のため、LASファイル座標系が同一座標系となるようにユーザーがフラグを切り替える必要があります。未記入、または、真偽値以外の値が入力された場合は、エラーメッセージを表示し、処理を中止します。
|4|	CityGMLFolderPath	| 文字列 | 入力CityGMLフォルダパスを指定します。未記入および指定されたフォルダが存在しない場合、フォルダ内にCityGMLファイルがない場合は無効とし、エラーメッセージを表示し、処理を中止します。
|5| TextureFolderPath	| 文字列 | 航空写真（原画像）の格納フォルダパスです。未記入および指定されたファイルが存在しない場合は無効とし、警告メッセージを表示し、テクスチャ貼付け処理を実施しません。
|6|	ExternalCalibElementPath | 文字列 |	外部標定パラメータのファイルパスです。未記入および指定されたファイルが存在しない場合は無効とし、警告メッセージを表示し、テクスチャ貼付け処理を実施しません。
|7|	RotateMatrixMode | 整数値 | テクスチャ貼付け処理において、ワールド座標からカメラ座標に変換する際に使用する回転行列Rの種類を切り替える設定値です。<br />モードの種類は以下2種類とします。<br />0:R=R_x (ω) R_y (ϕ) R_z (κ)<br />1:R=R_z (κ)R_y (ϕ)R_x (ω)<br/>未記入、または、0,1以外の値が入力された場合は、エラーメッセージを表示し、処理を中止します。
|8|	CameraInfoPath | 文字列 |	内部標定パラメータのファイルパスです。未記入および指定されたファイルが存在しない場合は無効とし、警告メッセージを表示し、テクスチャ貼付け処理を実施しません。
|9|	OutputFolderPath | 文字列 | 生成モデルの出力先フォルダパスです。指定されたフォルダ内に出力フォルダを作成し、作成したフォルダ内にCityGMLファイルとテクスチャ情報を出力します。テクスチャ情報は、入力CityGMLファイル名称(拡張子は除く)_appearanceフォルダに格納されます。
|10| OutputOBJ | 真偽値 | 生成モデルをCityGMLファイルとは別にOBJファイルとして出力するか否かを設定するフラグです。trueまたはfalseで値を指定します。未記入または、真偽値以外の値が入力された場合はエラーメッセージを表示し、処理を中止します。
|11| OutputTexture | 真偽値 | CityGMLファイルにテクスチャー情報を出力するか否かを設定するフラグです。trueまたはfalseで値を指定します。未記入の場合はtrueとなります。
|12| OutputLogFolderPath | 文字列 | ログのフォルダパスです。未記入または、存在しない場合は、本システムのPythonコードと同階層のログフォルダ“output_log”にログファイルを作成し、処理を中止します。
|13| DebugLogOutput | 真偽値 | デバッグレベルのログを出力するかどうかのフラグです。trueまたはfalseで値を指定します。未記入または、真偽値以外の値が入力された場合は、エラーメッセージを表示し、処理を中止します。
|14| PhaseConsistency	| 辞書 |	位相一貫性検査/補正処理用パラメータです。項目は位相一貫性検査/補正用設定パラメータ一覧を参照してください。

#### 選択パラメーター
| No |	キー名 |	値形式 | 説明 |
| -- | -- | -- | -- | 
|1| TargetCoordAreas | `Array<Array<Array<number>>>` | 緯度経度の領域を指定して建築物の対象を絞ります。入力しない場合、全ての建築物を対象とします


### LOD2建築物モデル自動作成開始
```
python3 AutoCreateLod2.py param.json
```


## 屋根面視認性向上ツールと壁面視認性向上ツール

### プロジェクト内相対パスへ移動 : ./tools/SuperResolution

### 依存ライブラリのインストール
```
pip install –r requirement.txt # 仮想環境の開始後
```

### 屋根面視認性向上用モデルの学習済みパラメーターをダウンロード
```
wget -O RoofSurface/checkpoint/iter_280000_conv.pth https://drive.google.com/file/d/1xBFAVgGeIGFsvMN6bG_Y9renLyNm46is/view?usp=drivesdk
```

### 壁面視認性向上用モデルの学習済みパラメーターをダウンロード
```
wget -O WallSurface/checkpoint/latest_net_G_A.pth https://github.com/realglobe-Inc/pytorch-CycleGAN-and-pix2pix/releases/download/bldg-lod2-tool-v2.0.0/latest_net_G_A.pth
```

### 屋根面視認性向上開始
```
python3 RoofSurface/CreateSuperResolution.py param.json
```

### 壁面視認性向上開始
```
python3 WallSurface/main.py param.json
```



## 画質向上ツール

### プロジェクト内相対パスへ移動 : ./tools/DeblurGANv2

### 依存ライブラリのインストール
```
pip install –r requirements.txt # 仮想環境の開始後
```

### 事前学習モデルの学習済みパラメーターをダウンロード
- `~/.cache/torch/hub/checkpoints` に `1inceptionresnetv2-520b38e4.pth`
```
wget -O ~/.cache/torch/hub/checkpoints/inceptionresnetv2-520b38e4.pth https://github.com/realglobe-Inc/DeblurGANv2/releases/download/v1.0.0/inceptionresnetv2-520b38e4.pth
```
- `checkpoints/fpn_inception.h5` に fpn_inception.h5
```
wget -O checkpoints/fpn_inception.h5 'https://docs.google.com/uc?export=download&id=1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR&confirm=t' #https://drive.google.com/open?id=1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR&authuser=0
```

### 画質向上開始
```
python3 predict.py param.json
```



## 画質のエッジシャープ化ツール

### プロジェクト内相対パスへ移動 : ./tools/UnsharpMask

### 依存ライブラリのインストール
```
pip install –r requirements.txt # 仮想環境の開始後
```

### 画質のエッジシャープ化ツール実行
```
python3 UnsharpMask.py param.json
```



## 解像度向上ツール

### プロジェクト内相対パスへ移動 : ./tools/Real-ESRGAN

### 依存ライブラリのインストール
```
# 仮想環境の開始後
pip install -r requirements.txt
python setup.py
```

### 事前学習モデルの学習済みパラメーターをダウンロード
```
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -P weights
```

### 解像度向上開始
```
# -g gpu-id
# -s 解像度向上(n倍)
# -n 事前学習モデル
# -i 入力画像
# -i 出力画像

# 4倍解像度向上
python3 inference_realesrgan.py -n RealESRGAN_x4plus -g 0 -s 4 -i input -o output

# 2倍解像度向上
python3 inference_realesrgan.py -n RealESRGAN_x2plus -g 0 -s 2 -i input -o output
```



## アトラス化ツール

### プロジェクト内相対パスへ移動 : ./tools/Atlas_Prot

### 依存ライブラリのインストール
```
pip install –r requirements.txt # 仮想環境の開始後
```

### アトラス化開始
```
python3 Atlas_Prot.py param.json
```


## 壁面視認性向上用モデルの学習済みパラメーター latest_net_G_A.pth の学習手順

### 壁面視認性向上ツールの学習コード clone
```
cd ~
git clone https://github.com/realglobe-Inc/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

### [python 仮想環境/開始](#python-仮想環境の設定方法)

### 学習データーダウンロード
```
wget -O datasets/d10.zip https://github.com/realglobe-Inc/pytorch-CycleGAN-and-pix2pix/releases/download/bldg-lod2-tool-v2.0.0/d10.zip
unzip datasets/d10.zip -d datasets/d10/
rm datasets/d10.zip
```

### CycleGAN B画像を選択
- B画像として高画質化済みの画像を選択する場合
```
cp -r datasets/d10/train_d10B_deblured/ datasets/d10/train_d10B/
```

- B画像として元の画像を選択する場合
```
cp -r datasets/d10/train_d10B_backup/ datasets/d10/train_d10B/
```

- 学習するデーターを追加する場合
  - datasets/d10/train_d10B_backup/ に B画像追加
  - datasets/d10/train_d10A/ に A画像追加
  - [画質向上ツール](#画質向上ツール)で `datasets/d10/train_d10B_backup/` の画質向上
  - 画質向上された画像を [画質のエッジシャープ化ツール](#画質のエッジシャープ化ツール)で `datasets/d10/train_d10B_backup/` ジシャープ化
  - ジシャープ化された画像を `datasets/d10/train_d10B/` にコピー


### 依存ライブラリのインストール
```
pip install –r requirements.txt # 仮想環境の開始後
```

### 学習開始
- 実行
```
python3 train.py --dataroot ./datasets/d10 --name CycleGAN_d10 --model cycle_gan --direction AtoB --phase train_d10 --save_epoch_freq 100 --n_epochs 500 --input_nc 3 --output_nc 3
```

- `checkpoints/CycleGAN_d10_old/latest_net_G_A.pth` が学習済みパラメーター

### テスト開始
- 実行
```
python3 test.py --dataroot ~/CycleGAN/datasets/d10 --name CycleGAN_d10 --model cycle_gan --direction AtoB --phase test_d10 --epoch  latest --input_nc 3 --output_nc 3
```

- `results` から結果確認
