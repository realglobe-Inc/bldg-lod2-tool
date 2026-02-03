import argparse
import json
import logging
import os
import shutil
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

import cv2
import yaml
from lxml import etree
from tqdm import tqdm

from cyclegan.dataset import DatasetDataLoader
from cyclegan.model.cyclegan_model import CycleGANModel
from cyclegan.util import util
from src.postprocessing import PostProcessing
from src.preprocessing import PreProcessing


def fix_relative_path(path):
    # pathlibで統一して簡潔にする
    return str(Path(path).expanduser().resolve())


def setup_logging(log_filename="debug.log", log_flag=False):
    # Create a logger
    logger = None

    # If DebugLogOutput is set to 'true', configure the logging
    if log_flag:
        logger = logging.getLogger(__name__)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)

        # Set the logging level
        logger.setLevel(logging.DEBUG)
        # Add the file handler to the logger
        logger.addHandler(file_handler)

    return logger


def write_log(log_root, action, filename=None):
    """
    Write a log entry to a log file.

    Parameters:
    - log_root: Path to the log file.
    - action: The action to log.
    - filename: Optional filename for additional information.
    """
    log_file = open(log_root, "a")

    if filename is not None:
        log_file.write(
            f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} : Execution of {filename}\n"
        )

    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} : {action}\n")
    log_file.close()


def check_path(path, cfg, logger):
    """
    Check if a path exists and raise an error if it doesn't.

    Parameters:
    - path: Path to check.
    - cfg: Configuration information.
    - logger: Logger for logging messages.
    """
    try:
        if not os.path.exists(path):
            raise ValueError(f"Error : {path} not found")
    except ValueError as ve:
        handle_error(ve, cfg, logger)


def handle_error(logger, error, log_flag):
    """
    Handle errors, log the message, and exit the program.

    Parameters:
    - error: The error that occurred.
    - param: Parameter Information.
    - logger: Logger for logging messages.
    """
    if log_flag:
        logger.error(f"{error}")
    print(f"Error: {error}")
    raise SystemExit(1)


def format_elapsed_time(process_time):
    """
    Format elapsed time into a human-readable string.

    Parameters:
    - process_time: Elapsed time in seconds.

    Returns:
    - Formatted elapsed time string.
    """
    hours, remainder = divmod(process_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds"


def check_error(param):
    """
    Check the validity of configuration information, log errors,
    and exit the program if there are any.

    Parameters:
    - param: Parameter Information.
    """
    try:
        log_root = None
        if not param.get("OutputLogDir"):
            log_root = Path("main_log.txt")
            bug_root = Path("debug.log")
        else:
            log_dir = os.path.join(
                param["OutputLogDir"], f"outputlog_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_root = Path(os.path.join(log_dir, "main_log.txt"))
            bug_root = Path(os.path.join(log_dir, "debug.log"))

        if not param.get("DebugLogOutput"):
            param["DebugLogOutput"] = "false"
        elif param.get("DebugLogOutput") not in ["true", "false"]:
            param["DebugLogOutput"] = "false"

        # Initialize the logger conditionally based on DebugLogOutput
        logger = setup_logging(bug_root, (param["DebugLogOutput"] == "true"))

        if not param.get("InputDir") or not param.get("OutputDir"):
            raise ValueError(
                "'InputDir' and 'OutputDir' must be specified in the JSON file."
            )

        if not param.get("InputDir"):
            param["Device"] = "cuda"
        elif param.get("Device") not in ["cuda", "cpu"]:
            param["Device"] = "cuda"

    except ValueError as ve:
        handle_error(logger, ve, (param["DebugLogOutput"] == "true"))

    except Exception as e:
        handle_error(logger, e, (param["DebugLogOutput"] == "true"))

    return log_root, logger


def get_original_texture_path(obj_file_path: Union[str, Path]):
    """
    Parse an OBJ file and its associated MTL file to extract textures
    matching the materials used in the OBJ file.

    Parameters:
    - obj_file_path (str or Path): Path to the OBJ file.

    Returns:
    - List of texture paths used in the OBJ file.
    """
    obj_file_path = Path(obj_file_path)
    if not obj_file_path.is_file():
        raise FileNotFoundError(f"OBJ file not found: {obj_file_path}")

    mtl_file_path = None
    used_materials = set()
    original_texture_path = None

    # Step 1: Parse the OBJ file to find the mtllib and used materials (usemtl)
    with obj_file_path.open("r") as obj_file:
        for line in obj_file:
            line = line.strip()
            if line.lower().startswith("mtllib "):  # Find the MTL file
                mtl_file_name = line.split(" ", 1)[1]
                mtl_file_path = obj_file_path.parent / mtl_file_name
            elif line.lower().startswith("usemtl "):  # Track used materials
                material_name = line.split(" ", 1)[1]
                used_materials.add(material_name)

    if not mtl_file_path or not mtl_file_path.is_file():
        raise FileNotFoundError(f"MTL file not found: {mtl_file_path}")

    # Step 2: Parse the MTL file to find matching materials and their textures
    current_material = None
    with mtl_file_path.open("r") as mtl_file:
        for line in mtl_file:
            line = line.strip()
            if line.lower().startswith("newmtl "):  # Start of a new material
                current_material = line.split(" ", 1)[1]
            elif (
                line.lower().startswith("map_kd ")
                and current_material in used_materials
            ):
                texture_relative_path = line.split(" ", 1)[1]
                texture_full_path = mtl_file_path.parent / texture_relative_path
                original_texture_path = texture_full_path.resolve()

    return original_texture_path


def get_texture_check_list(city_gml_path: Union[str, Path]):
    tree = ET.parse(city_gml_path)
    root = tree.getroot()
    namespaces = {"app": "http://www.opengis.net/citygml/appearance/2.0"}
    texture_check_list: dict[str, bool] = {}

    for image_uri in root.findall(".//app:imageURI", namespaces):
        texture_check_list[image_uri.text] = False

    return texture_check_list


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("param_file", type=Path)
    parser.add_argument("--cfg_file", type=Path, default="config.yml")
    args = parser.parse_args()

    # Load parameter information from JSON file
    with args.param_file.open("rt") as pf:
        param = json.load(pf)

    cfg_path = Path("src", args.cfg_file)
    with cfg_path.open("rt") as cf:
        cfg = yaml.safe_load(cf)

    # Enable Relative Path(.) and User Path(~)
    if param.get("InputDir"):
        param["InputDir"] = fix_relative_path(param["InputDir"])

    if param.get("OutputDir"):
        param["OutputDir"] = fix_relative_path(param["OutputDir"])

    if param.get("OutputLogDir"):
        param["OutputLogDir"] = fix_relative_path(param["OutputLogDir"])

    pixel_per_meter = 1 / float(param.get("MeterPerPixel", "0.16"))
    output_format = param.get("OutputFormat", "png").lower()

    # Check required fields in the configuration
    log_root, logger = check_error(param)

    # Create execution log
    start_time = time.time()
    with open(log_root, "w") as log_file:
        log_file.write(
            f"処理開始時刻 : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n"
        )
        log_file.write(f"指定パラメータ内容 : {json.dumps(param)}\n")
        log_file.write(f"各処理の詳細情報 : {json.dumps(cfg)}\n")

    cfg_process = cfg["processing"]
    processA_dir = Path(os.path.join(param["OutputDir"], "processA"))
    preprocessing = PreProcessing(
        logger=logger,
        overlap=cfg_process["overlap"],
        size=cfg_process["size"],
        pixel_per_meter=pixel_per_meter,
        z_threshold=cfg_process["z_threshold"],
        lower_limit=cfg_process["lower_limit"],
        upper_limit=cfg_process["upper_limit"],
    )

    cfg_cyclegan = cfg["cyclegan"]
    processB_dir = Path(os.path.join(param["OutputDir"], "processB"))
    dataset = DatasetDataLoader(cfg_cyclegan)
    model = CycleGANModel(cfg_cyclegan, param["Device"])
    model.setup(cfg_cyclegan)
    AtoB = cfg_cyclegan["direction"] == "AtoB"

    processC_dir = Path(os.path.join(param["OutputDir"], "processC"))
    postprocessing = PostProcessing(
        logger=logger,
        output_dir=processC_dir,
        overlap=cfg_process["overlap"],
        size=cfg_process["size"],
        z_threshold=cfg_process["z_threshold"],
    )

    city_gml_paths = Path(param["InputDir"]).iterdir()
    for city_gml_path in city_gml_paths:
        # Check cityGML
        if city_gml_path.suffix.lower() == ".gml":
            texture_check_list = get_texture_check_list(city_gml_path)

            input_obj_dir = Path(param["InputDir"]).joinpath(
                Path("obj", city_gml_path.stem)
            )
            output_obj_dir = Path(param["OutputDir"]).joinpath(
                Path("obj", city_gml_path.stem)
            )

            # Copy cityGML and Object Directories
            if output_obj_dir.is_dir():
                shutil.rmtree(output_obj_dir)
            shutil.copytree(input_obj_dir, output_obj_dir)
            shutil.copy(
                city_gml_path,
                Path(param["OutputDir"]).joinpath(Path(city_gml_path.name)),
            )

            isatty = sys.stdout.isatty()
            obj_files = [
                obj_file
                for obj_file in input_obj_dir.iterdir()
                if obj_file.suffix.lower() == ".obj"
            ]
            pbar = tqdm(
                total=len(obj_files),
                unit="file",
                leave=False,
                dynamic_ncols=isatty,
                disable=not isatty,
            )

            for obj_file in obj_files:
                pbar.set_description(f"{city_gml_path.stem}/{obj_file.name}")
                if not isatty:
                    print(f"Processing {city_gml_path.stem}/{obj_file.name}")

                try:
                    # Check object files
                    index = obj_file.name.replace(".", "_")
                    # Setting Sub Directories Paths
                    sub_processA_dir = processA_dir.joinpath(
                        Path(city_gml_path.stem, index)
                    )
                    sub_processB_dir = processB_dir.joinpath(
                        Path(city_gml_path.stem, index)
                    )
                    sub_processC_dir = processC_dir.joinpath(
                        Path(city_gml_path.stem, index)
                    )
                    # Creating Sub Directories Paths
                    if logger is not None:
                        sub_processA_dir.mkdir(exist_ok=True, parents=True)
                        sub_processB_dir.mkdir(exist_ok=True, parents=True)
                        sub_processC_dir.mkdir(exist_ok=True, parents=True)

                    # Check if the file is present
                    check_path(obj_file, param, logger)

                    original_texture_path = get_original_texture_path(obj_file)
                    if original_texture_path is None:
                        continue

                    # pre-processing
                    write_log(log_root, "変換対象壁面の抽出および正対化開始", obj_file)

                    try:
                        preprocess_log = preprocessing.main_step(
                            obj_file, sub_processA_dir
                        )
                    except Exception:
                        traceback.print_exc()
                        result = cv2.imread(str(original_texture_path))

                        # Saving output results
                        resolve_path_img = Path(original_texture_path)
                        relative_path_img = resolve_path_img.relative_to(
                            Path(param["InputDir"]).resolve()
                        )
                        output_path = Path(param["OutputDir"]).joinpath(
                            relative_path_img
                        )
                        output_path.parent.mkdir(exist_ok=True, parents=True)
                        cv2.imwrite(str(output_path), result)

                        if texture_check_list.get(str(relative_path_img)) is not None:
                            texture_check_list[str(relative_path_img)] = True

                        continue

                    # cyclegan processing
                    write_log(log_root, "壁面画像生成開始")
                    for num_iw, img_iw in enumerate(preprocess_log["output_images"]):
                        for num_ih, img_ih in enumerate(img_iw):
                            for num, img in enumerate(img_ih):
                                img = dataset.read_img(img["img"], img["path"])

                                model.set_input(img)
                                model.test()
                                visuals = (
                                    model.get_current_visuals()
                                )  # get image results
                                result = util.tensor2im(
                                    visuals["fake_B" if AtoB else "fake_A"]
                                )
                                if logger is not None:
                                    img_path = (
                                        model.get_image_paths()
                                    )  # get image paths
                                    util.save_image(
                                        result,
                                        os.path.join(
                                            sub_processB_dir,
                                            os.path.basename(str(img_path)),
                                        ),
                                    )

                                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                                preprocess_log["output_images"][num_iw][num_ih][num][
                                    "img"
                                ] = result

                    # post-processing
                    write_log(log_root, "アトラス化画像再構成開始")

                    try:
                        result = postprocessing.main_step(
                            preprocess_log, sub_processC_dir
                        )
                    except Exception:
                        traceback.print_exc()
                        if logger is not None:
                            logger.info(f"Failed PostProcessing:", obj_file)
                        else:
                            print(f"Failed PostProcessing:", obj_file)
                        result = cv2.imread(str(original_texture_path))

                    # Saving output results
                    resolve_path_img = Path(original_texture_path)
                    relative_path_img = resolve_path_img.relative_to(
                        Path(param["InputDir"]).resolve()
                    )
                    output_path = Path(param["OutputDir"]).joinpath(
                        os.path.splitext(relative_path_img)[0] + f".{output_format}"
                    )
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(output_path), result)

                    if texture_check_list.get(str(relative_path_img)) is not None:
                        texture_check_list[str(relative_path_img)] = True
                finally:
                    pbar.update(1)

            pbar.close()

            processed_count = 0
            for is_processed in texture_check_list.values():
                if is_processed:
                    processed_count += 1

            # 処理対象が0件の場合はスキップして次のGMLへ
            if processed_count == 0:
                print(
                    f"Warning: 処理対象のテクスチャ画像が見つかりませんでした: {city_gml_path.name}"
                )
                continue

            for texture_path, is_processed in texture_check_list.items():
                if not is_processed:
                    original_texture_path = Path(param["InputDir"]).joinpath(
                        texture_path
                    )
                    not_processed_texture = cv2.imread(str(original_texture_path))

                    output_path = Path(param["OutputDir"]).joinpath(
                        os.path.splitext(texture_path)[0] + f".{output_format}"
                    )
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(output_path), not_processed_texture)

    # 画像形式の変更
    for mtl_file in Path(param["OutputDir"]).rglob("*.mtl"):
        with open(mtl_file, "r") as file:
            lines = file.readlines()
        new_lines = []
        changed = False
        for line in lines:
            l = line.strip().lower()
            ext = l.split(".")[-1]
            if l.startswith("map_kd") and ext != output_format:
                line = line.rstrip().replace(f".{ext}", f".{output_format}")
                changed = True
            new_lines.append(line)
        if changed:
            with open(mtl_file, "w") as file:
                file.writelines(new_lines)

    # 画像形式の変更
    for gml_file in Path(param["OutputDir"]).rglob("*.gml"):
        # GMLファイルを解析
        tree = etree.parse(gml_file)
        root = tree.getroot()

        # app:surfaceDataMemberのnamespaceを取得
        namespaces = {"app": "http://www.opengis.net/citygml/appearance/2.0"}

        changed = False
        # app:Appearance要素を取得
        for appearance in root.findall(".//app:Appearance", namespaces):
            for surface_data_member in appearance.findall(
                "app:surfaceDataMember", namespaces
            ):
                parameterized_texture = surface_data_member.find(
                    "app:ParameterizedTexture", namespaces
                )
                if parameterized_texture is not None:
                    image_uri = parameterized_texture.find("app:imageURI", namespaces)
                    if image_uri is not None and not image_uri.text.endswith(
                        f".{output_format}"
                    ):
                        changed = True
                        image_uri.text = str(
                            Path(image_uri.text).with_suffix(f".{output_format}")
                        )
                    mime_type = parameterized_texture.find("app:mimeType", namespaces)
                    if mime_type is not None and not mime_type.text.endswith(
                        f"/{output_format}"
                    ):
                        changed = True
                        mime_type.text = f"image/{output_format}"

        if changed:
            # 結果を新しいファイルに保存 (XML宣言を含む)
            tree.write(
                gml_file, encoding="utf-8", xml_declaration=True, pretty_print=True
            )

    end_time = time.time()
    process_time = end_time - start_time
    with open(log_root, "a") as log_file:
        log_file.write(
            f"\n処理終了時刻 : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n"
        )
        log_file.write(f"トータル処理時間 : {format_elapsed_time(process_time)}\n")

    # debug log
    if logger is not None:
        logger.info(f"Total processing time: {format_elapsed_time(process_time)}")
