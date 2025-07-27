#!/bin/bash

# 이미지들이 있는 디렉토리
image_dir="../datas/demo"

# 편집 프롬프트 리스트
edit_prompts=(
  "Let it in a flood scene"
  "Let it in a spring scene"
  "Let it in an autumn scene"
  "Let it in a night scene"
  "Let it in a sunset scene"
  "Let it in a sunrise scene"
  "Let it in a snow scene"
  "Let it in a christmas scene"
  "Let it in a spring blossom tree scene"
)

# 이미지 디렉토리 내 모든 PNG 파일에 대해 반복
for img_path in "$image_dir"/*.png; do
  img_name=$(basename "$img_path" .png)  # 예: grey-houses

  # 프롬프트별 실행
  for edit_prompt in "${edit_prompts[@]}"; do
    keyword=$(echo "$edit_prompt" | sed 's/ /_/g')

    # 저장 폴더 생성 경로
    save_folder="../result/${img_name}_${keyword}_ablation_study_blockwise_control"

    echo "Running on image: $img_name with prompt: \"$edit_prompt\" → saving to: $save_folder"

    # 실행
    CUDA_VISIBLE_DEVICES=2 python ablation_study_blockwise_control.py \
      --save_folder "$save_folder" \
      --img_dir "$img_path" \
      --edit_prompt "$edit_prompt" \
      --num_inference_steps 28 \
      --structure_ideal_high 0.85
  done
done
