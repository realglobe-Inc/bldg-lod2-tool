import os
import json
import argparse
from glob import glob
from pathlib import Path

from PIL import Image, ImageFilter

def main(img_pattern='input/*', out_dir='output/'):
    def sorted_glob(pattern):
        return sorted(glob(pattern))

    imgs = sorted_glob(img_pattern)
    os.makedirs(out_dir, exist_ok=True)
    for img_path in imgs:
        # 画像を読み込む
        image = Image.open(img_path)

        # アンシャープマスクを適用
        sharpened_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

        # シャープ化した画像を保存
        basename = os.path.basename(img_path)
        save_path = os.path.join(out_dir, basename)
        sharpened_image.save(save_path)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("param_file", type=Path)
    args = parser.parse_args()

    # Load parameter information from JSON file
    with args.param_file.open("rt") as pf:
        param = json.load(pf)

    main(
        img_pattern=os.path.join(param['FilePath']['InputImagePath'], '*'),
        out_dir=param['FilePath']['OutputImagePath']
    )
