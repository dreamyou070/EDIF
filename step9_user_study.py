from diffusers import FluxKontextPipeline
import torch
from PIL import Image
import numpy as np
import random
from PIL import Image
import  os

def create_folder_with_full_permission(dir):
    os.makedirs(dir, exist_ok=True)                # 폴더 생성 (이미 있으면 무시)
    os.chmod(dir, 0o777)

def main() :


    print(f' step 1. call pipe')
    np.random.seed(42)
    MAX_SEED = np.iinfo(np.int32).max

    pipe = FluxKontextPipeline.from_pretrained(
        "/workspace/model/FluxKontext/models--black-forest-labs--FLUX.1-Kontext-dev/snapshots/af58063aa431f4d2bbc11ae46f57451d4416a170cd",
        #"black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16).to("cuda")

    print(f' step 2. call images')
    base_output_dir = "/data/CVPR_Public/Project_EDIF/Kontext_userstudy"
    create_folder_with_full_permission(base_output_dir)

    image_folder = '../user_study'
    edit_file = '../user_study/demo.json'
    import json
    with open(edit_file, 'r', encoding='utf-8') as f:
        contents = json.load(f)

    for data in contents:
        file = data['filename']
        prompt = data["scene_change"][0]
        key_prompt = data["scene_change"][1]
        name, ext = os.path.splitext(file)
        i_path = os.path.join(image_folder, file)
        img = Image.open(i_path).convert("RGB")
        output_name = f'{name}_{key_prompt}.png'
        output_dir = os.path.join(base_output_dir, output_name)

        guidance_scale = 2.5
        steps = 30

        start_image =img
        edited_img = pipe(
            image=start_image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            width=start_image.size[0],
            height=start_image.size[1],
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(42),
        ).images[0]

        edited_img.save(output_dir)

if __name__ == '__main__':
    main()