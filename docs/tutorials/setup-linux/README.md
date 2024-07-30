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
