FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# タイムゾーンの設定
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# 必要なパッケージのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    nano curl wget zip unzip libopencv-dev jq build-essential \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libffi-dev liblzma-dev git locales bc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8 ja_JP.UTF-8
RUN update-locale LANG=en_US.UTF-8

ENV HOME="/root"

# pyenv のインストール
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/bin:${PATH}"
RUN mkdir -p "${PYENV_ROOT}"
RUN git clone https://github.com/pyenv/pyenv.git "${PYENV_ROOT}" && \
    git clone https://github.com/pyenv/pyenv-virtualenv.git "${PYENV_ROOT}/plugins/pyenv-virtualenv" && \
    echo 'eval "$(pyenv init --path)"' >> "${HOME}/.bashrc" && \
    echo 'eval "$(pyenv init -)"' >> "${HOME}/.bashrc"

# Python 3.9.21 のインストールと設定
RUN eval "$(pyenv init --path)" && \
    pyenv install 3.9.21 && \
    pyenv global 3.9.21 && \
    pyenv rehash

# Python のパス設定
ENV PATH="${PYENV_ROOT}/shims:${PATH}"



########## LOD2建築物自動作成ツールのインストール ##########

# LOD2建築物自動作成ツールのフォルダーに移動
RUN mkdir -p /app
WORKDIR /app

# 必要なファイルをコピー
COPY requirements.txt .

# 必要なPythonライブラリをインストール
RUN python -m venv "$(basename $PWD)" && \
    . "$(basename $PWD)/bin/activate" && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    deactivate


# 学習済みモデルのダウンロード（ファイルがない場合のみ）
RUN mkdir -p src/createmodel/data && \
    test -f src/createmodel/data/classifier_parameter.pkl || \
    wget -q 'https://github.com/realglobe-Inc/bldg-lod2-tool/releases/download/PretrainedModels-1.0/classifier_parameter.pkl' \
      -O src/createmodel/data/classifier_parameter.pkl && \
    test -f src/createmodel/data/roof_edge_detection_parameter.pth || \
    wget -q 'https://github.com/realglobe-Inc/bldg-lod2-tool/releases/download/PretrainedModels-1.0/roof_edge_detection_parameter.pth' \
      -O src/createmodel/data/roof_edge_detection_parameter.pth && \
    test -f src/createmodel/data/balcony_segmentation_parameter.pkl || \
    wget -q 'https://github.com/realglobe-Inc/bldg-lod2-tool/releases/download/PretrainedModels-1.0/balcony_segmentation_parameter.pkl' \
      -O src/createmodel/data/balcony_segmentation_parameter.pkl



########## 壁面視認性向上ツールのインストール ##########

# 壁面視認性向上ツールのフォルダーに移動
RUN mkdir -p /app/tools/SuperResolution/WallSurface
WORKDIR /app/tools/SuperResolution/WallSurface

# 必要なファイルをコピー
COPY tools/SuperResolution/WallSurface/checkpoint checkpoint
COPY tools/SuperResolution/WallSurface/requirements.txt .

# 必要なPythonライブラリをインストール
RUN python -m venv "$(basename $PWD)" && \
    . "$(basename $PWD)/bin/activate" && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    deactivate

# 学習済みモデルのダウンロード（ファイルがない場合のみ）
RUN test -f checkpoint/latest_net_G_A.pth || \
    wget -q 'https://github.com/realglobe-Inc/pytorch-CycleGAN-and-pix2pix/releases/download/bldg-lod2-tool-v2.0.0/latest_net_G_A.pth' \
      -O checkpoint/latest_net_G_A.pth



########## テクスチャ鮮明化ツールのインストール ##########

# テクスチャ鮮明化ツールのフォルダーに移動
RUN mkdir -p /app/tools/DeblurGANv2
WORKDIR /app/tools/DeblurGANv2

# 必要なファイルをコピー
COPY tools/DeblurGANv2/checkpoints checkpoints
COPY tools/DeblurGANv2/requirements.txt .

# 必要なPythonライブラリをインストール
RUN python -m venv "$(basename $PWD)" && \
    . "$(basename $PWD)/bin/activate" && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    deactivate

# 学習済みモデルのダウンロード（ファイルがない場合のみ）
RUN mkdir -p "${HOME}/.cache/torch/hub/checkpoints" && \
    test -f "${HOME}/.cache/torch/hub/checkpoints/inceptionresnetv2-520b38e4.pth" || \
    wget -q 'https://github.com/realglobe-Inc/DeblurGANv2/releases/download/v1.0.0/inceptionresnetv2-520b38e4.pth' \
      -O "${HOME}/.cache/torch/hub/checkpoints/inceptionresnetv2-520b38e4.pth" && \
    test -f checkpoints/fpn_inception.h5 || \
    wget -q 'https://github.com/realglobe-Inc/DeblurGANv2/releases/download/v1.0.0/fpn_inception.h5' \
      -O checkpoints/fpn_inception.h5



########## テクスチャシャープ化ツールのインストール ##########
# テクスチャシャープ化ツールはきれいにならないため使わない



########## テクスチャ解像度向上ツールのインストール ##########

# テクスチャ解像度向上ツールのフォルダーに移動
RUN mkdir -p /app/tools/Real-ESRGAN
WORKDIR /app/tools/Real-ESRGAN

# 必要なファイルをコピー
COPY tools/Real-ESRGAN/weights weights
COPY tools/Real-ESRGAN/realesrgan realesrgan
COPY tools/Real-ESRGAN/setup.py .
COPY tools/Real-ESRGAN/VERSION .
COPY tools/Real-ESRGAN/README.md .
COPY tools/Real-ESRGAN/requirements.txt .

# 必要なPythonライブラリをインストール
RUN python -m venv "$(basename $PWD)" && \
    . "$(basename $PWD)/bin/activate" && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python setup.py develop && \
    deactivate

# 学習済みモデルのダウンロード（ファイルがない場合のみ）
RUN test weights/RealESRGAN_x4plus.pth || \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
      -O weights/RealESRGAN_x4plus.pth



########## テクスチャアトラス化ツールのインストール ##########
# テクスチャアトラス化ツールはうまく動かないため使わない



########## テクスチャ正対化ツール等のインストール ##########

# テクスチャ正対化ツールのフォルダーに移動
RUN mkdir -p /app/tools/misc
WORKDIR /app/tools/misc

# 必要なファイルをコピー
COPY tools/misc/requirements.txt .

# 必要なPythonライブラリをインストール
RUN python -m venv "$(basename $PWD)" && \
    . "$(basename $PWD)/bin/activate" && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    deactivate


########################################################################
########## ビルド速度向上のため、この下は頻繁に変更されるファイルの処理 ##########
########################################################################



########## LOD2建築物自動作成ツールの頻繁に変更されるファイル ##########

# LOD2建築物自動作成ツールのフォルダーに移動
WORKDIR /app

# 必要なファイルをコピー
COPY src src
COPY AutoCreateLod2.py .



########## 壁面視認性向上ツールの頻繁に変更されるファイル ##########

# 壁面視認性向上ツールのフォルダーに移動
WORKDIR /app/tools/SuperResolution/WallSurface

# 必要なファイルをコピー
COPY tools/SuperResolution/WallSurface/src src
COPY tools/SuperResolution/WallSurface/cyclegan cyclegan
COPY tools/SuperResolution/WallSurface/main.py .



########## テクスチャ鮮明化ツールの頻繁に変更されるファイル ##########

# テクスチャ鮮明化ツールのフォルダーに移動
WORKDIR /app/tools/DeblurGANv2

# 必要なファイルをコピー
COPY tools/DeblurGANv2/models models
COPY tools/DeblurGANv2/config config
COPY tools/DeblurGANv2/predict.py .
COPY tools/DeblurGANv2/aug.py .



########## テクスチャシャープ化ツールの頻繁に変更されるファイル ##########
# テクスチャシャープ化ツールはきれいにならないため使わない



########## テクスチャ解像度向上ツールの頻繁に変更されるファイル ##########

# テクスチャ解像度向上ツールのフォルダーに移動
WORKDIR /app/tools/Real-ESRGAN

# 必要なファイルをコピー
COPY tools/Real-ESRGAN/inference_realesrgan.py .



########## テクスチャアトラス化ツールの頻繁に変更されるファイル ##########
# テクスチャアトラス化ツールはうまく動かないため使わない



########## テクスチャ正対化ツール等の頻繁に変更されるファイル ##########

# テクスチャ正対化ツールのフォルダーに移動
WORKDIR /app/tools/misc

# 必要なファイルをコピー
COPY tools/misc/change_texture_image_ext_in_gml.py .
COPY tools/misc/rectify_texture_image.py .



########## 実行関連設定 ##########

# デフォルト実行パス
WORKDIR /app

# 実行ファイルをコピー
COPY process.sh .
