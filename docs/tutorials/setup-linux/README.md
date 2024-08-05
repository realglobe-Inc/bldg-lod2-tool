# AWS EC2 Ubuntu 20.04 での環境構築のガイド

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
wget https://drive.google.com/file/d/1hs-DT4Y0ZtjdV9kJ438lvAPpJcfz_dE_/view?usp=drive_link 	classifier_parameter.pkl
mv classifier_parameter.pkl src/createmodel/data/
```

### 屋根線検出用モデルの学習済みパラメーターをダウンロード
```
wget https://drive.google.com/file/d/1QqxfS05a4T1_IdrzYle3iuBXjuyqFz-u/view?usp=drive_link roof_edge_detection_parameter.pth
mv roof_edge_detection_parameter.pth src/createmodel/data/
```

### バルコニー検出用モデルの学習済みパラメーターをダウンロード
```
wget https://drive.google.com/file/d/1MINHffIvcooDOrQq3E4mBvdsgWUfzIi5/view?usp=drive_link roof_edge_detection_parameter.pth
mv roof_edge_detection_parameter.pth src/createmodel/data/
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

### LOD2建築物モデル自動作成パラメーター
param.json
```
{
  "LasCoordinateSystem": 9,
  "DsmFolderPath": "/home/ubuntu/AutoCreateLod2_tutorial/LOD2Creator_tutorial/dataset/DSM",
  "LasSwapXY": false,
  "CityGMLFolderPath": "/home/ubuntu/AutoCreateLod2_tutorial/LOD2Creator_tutorial/dataset/CityGML",
  "TextureFolderPath": "/home/ubuntu/AutoCreateLod2_tutorial/LOD2Creator_tutorial/dataset/RawImage",
  "RotateMatrixMode": 0,
  "ExternalCalibElementPath": "/home/ubuntu/AutoCreateLod2_tutorial/LOD2Creator_tutorial/dataset/ExCalib/ExCalib.txt",
  "CameraInfoPath": "/home/ubuntu/AutoCreateLod2_tutorial/LOD2Creator_tutorial/dataset/CamInfo/CamInfo.txt",
  "OutputFolderPath": "/home/ubuntu/AutoCreateLod2_tutorial/output",
  "OutputOBJ": false,
  "OutputTexture": true,
  "OutputLogFolderPath": "/home/ubuntu/AutoCreateLod2_tutorial/output",
  "DebugLogOutput": false,
  "PhaseConsistency": {
    "DeleteErrorObject": true,
    "NonPlaneThickness": 0.05,
    "NonPlaneAngle": 15
  },
  "TargetGeoArea" : [[35, 139], [36, 140]]
}
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
|1| TargetGeoArea | `Array<Array<number>>` | 緯度経度の領域を指定して建築物の対象を絞ります。入力しない場合、全ての建築物を対象とします


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
wget https://drive.google.com/file/d/1xBFAVgGeIGFsvMN6bG_Y9renLyNm46is/view?usp=drivesdk iter_280000_conv.pth
mv iter_280000_conv.pth RoofSurface/checkpoint/
```

### 壁面視認性向上用モデルの学習済みパラメーターをダウンロード
```
wget https://drive.google.com/file/d/14tsr1r1s6aI6fm-cX7ZfcGr-56SdiTid/view?usp=drivesdk latest_net_G_A.pth
mv latest_net_G_A.pth WallSurface/checkpoint/
```

### 屋根面視認性向上開始
```
python3 RoofSurface/CreateSuperResolution.py param.json
```

### 壁面視認性向上開始
```
python3 SuperResolution/WallSurface/main.py param.json
```



## 画質向上ツール

### プロジェクト内相対パスへ移動 : ./tools/DeblurGANv2

### 依存ライブラリのインストール
```
pip install –r requirements.txt # 仮想環境の開始後
```

### 事前学習モデルの学習済みパラメーターをダウンロード
- [ブラウザからダウンロード(wget でダウンロードするエラー発生)](http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth)
- scp で inceptionresnetv2-520b38e4.pth をインスタンスにコピー
```
scp ~/Downloads/inceptionresnetv2-520b38e4.pth ubuntu@xxx.xxx.xxx.xxx:~/.cache/torch/hub/checkpoints/
```
- インスタンス内部で fpn_inception.h5
```
wget -O checkpoints/fpn_inception.h5 'https://docs.google.com/uc?export=download&id=1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR&confirm=t' #https://drive.google.com/open?id=1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR&authuser=0
```

### 画質向上開始
```
python3 predict.py param.json
```


## 解像度向上ツール

### プロジェクト内相対パスへ移動 : ./tools/Real-ESRGAN

### 依存ライブラリのインストール
```
python setup.py # 仮想環境の開始後
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
