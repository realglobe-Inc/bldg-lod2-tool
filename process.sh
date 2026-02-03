#!/bin/sh

set -e

########### Docker コンテナ内部で実行されるスクリプトです ###########
# OUTPUT_DIR:
#     出力ファイルや中間ファイルを保存するディレクトリパス。
#     最終出力は ${OUTPUT_DIR}/output_result 。
# BLDG_LOD2_TOOL_PARAM_FILE:
#     LOD2自動モデリングの設定ファイルパス。
#     これか BLDG_LOD2_TOOL_PARAM か INPUT_DIR、LAS_COORDINATE_SYSTEM、
#     OUTPUT_TEXTURE_ENABLED の組で自動モデリング部分を制御する。
#     優先度は BLDG_LOD2_TOOL_PARAM_FILE > BLDG_LOD2_TOOL_PARAM >
#     INPUT_DIR、LAS_COORDINATE_SYSTEM、OUTPUT_TEXTURE_ENABLED の組。
#     BLDG_LOD2_TOOL_PARAM_FILE も BLDG_LOD2_TOOL_PARAM も
#     LAS_COORDINATE_SYSTEM も空の場合はLOD2自動モデリングはスキップする。
# BLDG_LOD2_TOOL_PARAM:
#     LOD2自動モデリングの設定。
#     JSONをそのまま入れる。
# INPUT_DIR:
#     LAS_COORDINATE_SYSTEM が空でない場合はLOD2自動モデリングの入力ディレクトリ。
#     LOD2自動モデリングがスキップされた場合はテクスチャ高精度化の入力ディレクトリ。
# LAS_COORDINATE_SYSTEM:
#     LOD2自動モデリングの際の座標系。
# OUTPUT_TEXTURE_ENABLED:
#     LOD2自動モデリングの際にテクスチャも作成するか。
# METER_PER_PIXEL:
#     1ピクセルが何mに相当するか。
#     デフォルト0.25。
# METER_PER_TEXTURE_PIXEL:
#     テクスチャ画像の1ピクセルを何mに相当させるか。
#     デフォルト0.25。


output_dir="$(realpath -sm "${OUTPUT_DIR:?}")"

if [ -n "${BLDG_LOD2_TOOL_PARAM_FILE}" ]; then
  bldg_lod2_tool_param_file="$(realpath "${BLDG_LOD2_TOOL_PARAM_FILE}")"
fi
bldg_lod2_tool_param="${BLDG_LOD2_TOOL_PARAM}"
if [ -n "${INPUT_DIR}" ]; then
  input_dir="$(realpath "${INPUT_DIR}")"
fi
las_coordinate_system="${LAS_COORDINATE_SYSTEM}"
output_texture_enabled="${OUTPUT_TEXTURE_ENABLED:-false}"

meter_per_pixel="${METER_PER_PIXEL:-0.25}"
meter_per_texture_pixel="${METER_PER_TEXTURE_PIXEL:-"${meter_per_pixel:-0.25}"}"

echo "output_dir: ${output_dir}"
echo "bldg_lod2_tool_param_file: ${bldg_lod2_tool_param_file}"
printf 'bldg_lod2_tool_param: %s\n' "${bldg_lod2_tool_param}"
echo "input_dir: ${input_dir}"
echo "las_coordinate_system: ${las_coordinate_system}"
echo "output_texture_enabled ${output_texture_enabled}"
echo "meter_per_texture_pixel: ${meter_per_texture_pixel}"

project_dir="$(realpath "$(dirname "$0")")"



########## LOD2建築物自動作成ツール ##########

# LOD2建築物自動作成ツールのフォルダーに移動
cd "${project_dir}"

skip_bldg_lod2_tool=false
if [ -z "${bldg_lod2_tool_param_file}" ]; then
  bldg_lod2_tool_param_file="$(mktemp --suffix .json)"
  if [ -n "${bldg_lod2_tool_param}" ]; then
    printf '%s' "${bldg_lod2_tool_param}" > "${bldg_lod2_tool_param_file}"
  elif [ -z "${las_coordinate_system}" ]; then
    skip_bldg_lod2_tool=true
  elif [ -n "${input_dir}" ]; then
    cat <<EOF > "${bldg_lod2_tool_param_file}"
{
  "TextureFolderPath": "${input_dir}/RawImage",
  "ExternalCalibElementPath": "${input_dir}/ExCalib/ExCalib.txt",
  "CameraInfoPath": "${input_dir}/CamInfo/CamInfo.txt",
  "DsmFolderPath": "${input_dir}/DSM",
  "CityGMLFolderPath": "${input_dir}/CityGML",
  "LasCoordinateSystem": ${las_coordinate_system},
  "LasSwapXY": false,
  "RotateMatrixMode": 0,
  "OutputFolderPath": "${output_dir}",
  "OutputOBJ": true,
  "OutputTexture": ${output_texture_enabled},
  "OutputCityGML": true,
  "OutputLogFolderPath": "${output_dir}",
  "DebugLogOutput": true,
  "PhaseConsistency": {
    "DeleteErrorObject": true,
    "NonPlaneThickness": 0.05,
    "NonPlaneAngle": 15
  },
  "DebugMode": false,
  "TargetCoordAreas": null,
  "TargetBuildingIds": null,
  "TextureOutputWidthMax": 4096,
  "TextureOutputHeightMax": 4096,
  "TextureImageFormat": "png",
  "GridSize": ${meter_per_pixel},
}
EOF
  else
    echo 'LOD2建築物の自動作成には BLDG_LOD2_TOOL_PARAM_FILE か BLDG_LOD2_TOOL_PARAM か INPUT_DIR が必要です' 1>&2
    exit 1
  fi
fi

if ! "${skip_bldg_lod2_tool}"; then
  echo '########## LOD2建築物自動作成ツール ##########'

  output_texture_enabled="$(jq -r '.OutputTexture' "${bldg_lod2_tool_param_file}")"
  city_gml_dir_name="$(basename "$(jq -r '.CityGMLFolderPath' "${bldg_lod2_tool_param_file}")")"
  output_bldg_lod2_tool_dir_path="$(realpath -sm "$(jq -r '.OutputFolderPath' "${bldg_lod2_tool_param_file}")")"

  . "./$(basename $PWD)/bin/activate"
  python AutoCreateLod2.py "${bldg_lod2_tool_param_file}"
  deactivate

  # 最新のフォルダを取得
  output_bldg_lod2_tool_path="${output_dir}/output_bldg_lod2_tool"
  latest_folder="$(ls -t "${output_bldg_lod2_tool_dir_path}" | grep -E "^${city_gml_dir_name}_[0-9]{8}_[0-9]{4}$" | head -n 1)"
  if [ -n "${latest_folder}" ]; then
    cd "${output_dir}"
    rm -f ./output_bldg_lod2_tool
    ln -s "${output_bldg_lod2_tool_dir_path}/${latest_folder}" ./output_bldg_lod2_tool
  else
    echo '最新のフォルダが見つかりませんでした。' 1>&2
    exit 1
  fi



  if ! "${output_texture_enabled}"; then
    # テクスチャつくらないなら終わり
    output_result_path="${output_dir}/output_result"
    rm -rf "${output_result_path}/"
    cp -rL "${output_bldg_lod2_tool_path}" "${output_result_path}"

    echo "最終結果 : ${output_result_path}"

    mv "${output_result_path}" ./output

    output_gml=${OUTPUT_GML:-"false"}
    output_obj=${OUTPUT_OBJ:-"false"}
    output_tex=${OUTPUT_TEX:-"false"}

    if [ "$output_gml" = "false" ]; then
      rm -f output/*.gml
    fi

    if [ "$output_obj" = "false" ]; then
      rm -fr output/obj
    fi

    if [ "$output_tex" = "false" ]; then
      rm -rf output/*_appearance
    fi

    zip -r /opt/ml/processing/output/result.zip ./output

    # デバッグ用に全ファイルをzip化
    zip -r /opt/ml/processing/output/all.zip "${output_dir}"

    touch /opt/ml/processing/output/job_completed.txt

    exit 0
  fi
elif [ -n "${input_dir}" ]; then
  output_bldg_lod2_tool_path="${input_dir}"
else
  echo 'テクスチャ高精度化には INPUT_DIR が必要です' 1>&2
  exit 1
fi


########## 正対化ツール ##########

echo '########## 正対化ツール ##########'

# 正対化ツールのフォルダーに移動
cd "${project_dir}/tools/misc"

output_rectify_path="${output_dir}/output_rectify"
rm -rf "${output_rectify_path}/"*

. "./$(basename $PWD)/bin/activate"
python rectify_texture_image.py -i "${output_bldg_lod2_tool_path}" -o "${output_rectify_path}" \
 --format png --meter-per-pixel "${meter_per_texture_pixel}"
deactivate



########## 壁面視認性向上ツール ##########

echo '########## 壁面視認性向上ツール ##########'

# 壁面視認性向上ツールのフォルダーに移動
cd "${project_dir}/tools/SuperResolution/WallSurface"

output_wall_path="${output_dir}/output_wall"
rm -rf "${output_wall_path}/"*

wall_param_file="$(mktemp --suffix .json)"
cat <<EOF > "${wall_param_file}"
{
  "InputDir": "${output_rectify_path}",
  "OutputDir": "${output_wall_path}",
  "Device": "cuda",
  "OutputLogDir": "${output_dir}/log_output_wall",
  "DebugLogOutput": "false",
  "MeterPerPixel": "${meter_per_texture_pixel}",
  "OutputFormat": "png"
}
EOF

. "./$(basename $PWD)/bin/activate"
python main.py "${wall_param_file}"
deactivate



########## テクスチャ解像度向上ツール ##########

echo '########## テクスチャ解像度向上ツール ##########'

# テクスチャ解像度向上ツールのフォルダーに移動
cd "${project_dir}/tools/Real-ESRGAN"

output_esrgan_path="${output_dir}/output_esrgan"
rm -rf "${output_esrgan_path}/*"

. "./$(basename $PWD)/bin/activate"
python inference_realesrgan.py \
  -n RealESRGAN_x4plus -g 0 -s 4 --tile 1024 \
  -i "${output_wall_path}" -o "${output_esrgan_path}" \
  --input-ext png --ext png
deactivate


copy_misc() {
  input_image_path="${1}"
  input_misc_path="${2}"
  output_result_path="${3}"
  output_format="${4}"
  if [ "${input_image_path}" != "${output_result_path}" ]; then
    rm -rf "${output_result_path}"
    cp -r "${input_image_path}" "${output_result_path}"
  fi

  # テクスチャ画像以外のファイルをコピー
  for appearance_dir in "${output_result_path}/"*_appearance; do
    area="$(basename -s '_appearance' "${appearance_dir}")"

    if [ -f "${input_misc_path}/${area}.gml" ]; then
      area_label="${area}"
    else
      for gml_file in "${input_misc_path}/${area}_"*".gml"; do
        area_label=$(basename -s .gml "${gml_file}")
        break
      done
    fi

    # gmlファイルをコピー
    cp -n "${input_misc_path}/${area_label}.gml" "${output_result_path}/"

    # objファイルをコピー
    obj_dir_path="${output_result_path}/obj/${area_label}"
    mkdir -p "${obj_dir_path}"
    for texture_file in "${appearance_dir}/"*."${output_format}"; do
      bldg_id="$(basename -s ."${output_format}" "${texture_file}")"
      cp -n "${input_misc_path}/obj/${area_label}/${bldg_id}.obj" "${obj_dir_path}/"
    done

    # mtlファイル作成
    mtl_file_path="${obj_dir_path}/${area_label}.mtl"
    rm -f "${mtl_file_path}"
    for texture_file in "${appearance_dir}/"*."${output_format}"; do
      bldg_id="$(basename -s ."${output_format}" "${texture_file}")"
      printf '%s\n\n' "newmtl ${bldg_id}" >> "${mtl_file_path}"
      printf '%s\n\n' "map_Kd $(realpath --relative-to "${obj_dir_path}" "${texture_file}")" >> "${mtl_file_path}"
    done
  done

  # gmlファイルの中で変更
  cd "${project_dir}/tools/misc"
  . "./$(basename $PWD)/bin/activate"
  for gml_file in $(find "${output_result_path}" -name '*.gml'); do
    python change_texture_image_ext_in_gml.py -i "${gml_file}" -o "${gml_file}" --ext "${output_format}"
  done
  deactivate
}

copy_misc "${output_esrgan_path}" "${output_wall_path}" "${output_esrgan_path}" png



########## 壁面視認性向上ツール（2度掛け） ##########

echo '########## 壁面視認性向上ツール（2度掛け） ##########'

# 壁面視認性向上ツールのフォルダーに移動
cd "${project_dir}/tools/SuperResolution/WallSurface"

output_wall_path2="${output_dir}/output_wall2"
rm -rf "${output_wall_path2}/"*

wall_param_file2="$(mktemp --suffix .json)"
meter_per_texture_pixel2=$(echo "scale=8; ${meter_per_texture_pixel} / 4" | bc | sed 's/^\./0./; s/\(\.[0-9]*[1-9]\)0*$/\1/; s/\.0*$//')
cat <<EOF > "${wall_param_file2}"
{
  "InputDir": "${output_esrgan_path}",
  "OutputDir": "${output_wall_path2}",
  "Device": "cuda",
  "OutputLogDir": "${output_dir}/log_output_wall2",
  "DebugLogOutput": "false",
  "MeterPerPixel": "${meter_per_texture_pixel2}",
  "OutputFormat": "png"
}
EOF

. "./$(basename $PWD)/bin/activate"
python main.py "${wall_param_file2}"
deactivate



########## テクスチャ鮮明化ツール ##########

echo '########## テクスチャ鮮明化ツール ##########'

# テクスチャ鮮明化ツールのフォルダーに移動
cd "${project_dir}/tools/DeblurGANv2"

output_deblurgan_path="${output_dir}/output_deblurgan"
rm -rf "${output_deblurgan_path}/"*

. "./$(basename $PWD)/bin/activate"
python predict.py \
  -c checkpoints/fpn_inception.h5 \
  -i "${output_wall_path2}" -o "${output_deblurgan_path}" \
  --input-format png --output-format jpg
deactivate



########## テクスチャシャープ化ツール ##########
# テクスチャシャープ化ツールはきれいにならないため使わない

########## テクスチャアトラス化ツール ##########
# テクスチャアトラス化ツールはうまく動かないため使わない



########## 最終結果フォルダー ##########

output_path="${output_dir}/output_result"
copy_misc "${output_deblurgan_path}" "${output_wall_path2}" "${output_path}" jpg

echo "最終結果 : ${output_path}"

mv "${output_path}" ./output

output_gml=${OUTPUT_GML:-"false"}
output_obj=${OUTPUT_OBJ:-"false"}
output_tex=${OUTPUT_TEX:-"false"}

if [ "$output_gml" = "false" ]; then
  rm -f output/*.gml
fi

if [ "$output_obj" = "false" ]; then
  rm -fr output/obj
fi

if [ "$output_tex" = "false" ]; then
  rm -rf output/*_appearance
fi

zip -r /opt/ml/processing/output/result.zip ./output

# デバッグ用に全ファイルをzip化
zip -r /opt/ml/processing/output/all.zip "${output_dir}"

touch /opt/ml/processing/output/job_completed.txt