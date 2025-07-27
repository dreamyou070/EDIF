import torch, json
from pipelines.pipelines_imageedit import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image
from diffusers.models.embeddings import apply_rotary_emb
import torch
from PIL import Image
import torch
import numpy as np
from PIL import Image
import cv2
import argparse
import os
import torch
from pipelines.pipelines_imageedit import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image
from types import MethodType
from diffusers.models.embeddings import apply_rotary_emb
import torch
from PIL import Image
import math
import torch
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from skimage.metrics import structural_similarity as ssim
# export HF_HOME=/data/CVPR_Public/huggingface

def create_folder_with_full_permissions(path: str):
    os.makedirs(path, exist_ok=True)
    os.chmod(path, 0o777)

# 1. image_alpha : min (0.6) max()

def image_comparison(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compare two images using SSIM (Structural Similarity Index).
    Both images must be grayscale and have the same shape.
    """
    # Resize image2 to match image1
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)

    # SSIM expects float images in [0, 255] or normalized [0, 1]
    score = ssim(image1, image2, data_range=255)
    return score

def image_comparison_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """ image_similarity_sift
    클 수록 유사한 값이다. 값의 범위 0 ~ 1
    Compare two images using SIFT feature matching.
    Returns a similarity score: (number of good matches) / (min number of keypoints).
    """

    # 이미지를 그레이스케일로 변환
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY) if len(image1.shape) == 3 else image1
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY) if len(image2.shape) == 3 else image2

    # SIFT 디스크립터 생성
    sift = cv2.SIFT_create()

    # 특징점 및 디스크립터 추출
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return 0.0  # 특징점이 없는 경우

    # BFMatcher 생성 및 매칭
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test 적용
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # 유사도 계산: 좋은 매칭 수 / 가능한 최소 디스크립터 수
    min_des = min(len(des1), len(des2))
    if min_des == 0:
        return 0.0

    similarity = len(good_matches) / min_des
    return similarity

class Controller():

    def __init__(self):
        self.attention_dict = {}
        self.layer_timenum_dict = {}
        self.skiptime = None
        self.original_forward = ""
        self.entangled_value_dict = {}
        self.original_forward_dict = {}
        self.alpha_dict = {}
        self.text_strength_dict = {}
        self.img_alpha_dict = {}
        self.txt_alpha_dict = {}
        self.state = ''
        self.text_len = 0
        self.source_img_strength_dict = {}
        self.layerwise_text_strength = {}
        self.infer_time_dict = {}

    def add_infer_time(self, block_idx):
        if block_idx not in self.infer_time_dict:
            self.infer_time_dict[block_idx] = 1
        self.infer_time_dict[block_idx] = self.infer_time_dict[block_idx] + 1

    def save_text_strength(self, save_layer_name, text_strength):
        self.layerwise_text_strength[save_layer_name] = text_strength

    def set_state(self, state):
        self.state = state

    def save_value(self, block_id, text_strength, source_strength):
        self.text_strength_dict[block_id] = text_strength
        self.source_img_strength_dict[block_id] = source_strength

    def set_alpha(self, attn_idx, txt_alpha, img_alpha):
        self.txt_alpha_dict[attn_idx] = txt_alpha
        self.img_alpha_dict[attn_idx] = img_alpha

    def reset(self):
        self.layer_timenum_dict = {}
        self.attention_dict = {}
        self.txt_alpha_dict = {}
        self.skiptime = None
        self.original_forward = ""
        self.entangled_value_dict = {}
        self.original_forward_dict = {}
        self.alpha_dict = {}
        self.text_strength_dict = {}
        self.state = ''
        self.text_len = 0
        self.source_img_strength_dict = {}
        self.layerwise_text_strength = {}
        self.img_alpha_dict = {}
        self.infer_time_dict = {}
    def save_attn_map(self, timestep, attn_map):
        if timestep not in self.attention_dict:
            self.attention_dict[timestep] = []
        self.attention_dict[timestep].append(attn_map)

    def save_entangled_value(self, save_layer_name, entangled_value):
        self.entangled_value_dict[save_layer_name] = entangled_value

    def save_forward(self, block_idx, forward_fn):
        self.original_forward_dict[block_idx] = forward_fn

QIB = [1, 17, 27, 32, 56]
SIB = [18, 19, 31, 41]
QTB = [1]
ETB = [1, 18]

def main(args):

    print(f' step 1. Call VLM')
    device = 'cuda'

    print(f' step 2. Call Flux Pipe')
    pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev",
                                               torch_dtype=torch.bfloat16)
    pipe.to('cuda')
    flux_transformer = pipe.transformer
    print(f' (2.1) register new function')

    print(f' step 3. save folder')
    save_folder_base = '/data/CVPR_Public/Project_EDIF/Base'
    create_folder_with_full_permissions(save_folder_base)
    print(f' step 2. Edit with Total')
    def ca_forward(attn_module, layer_idx, controller):
        def forward(hidden_states: torch.FloatTensor,
                    encoder_hidden_states: torch.FloatTensor = None,
                    attention_mask=None,
                    image_rotary_emb=None,
                    **kwargs) -> torch.FloatTensor:

            attn = attn_module
            is_cross = encoder_hidden_states is not None

            # alpha 조정
            txt_alpha = controller.txt_alpha_dict.get(layer_idx, 1.0)
            img_alpha = controller.img_alpha_dict.get(layer_idx, 1.0)

            if is_cross:
                encoder_hidden_states *= txt_alpha
                total_len = hidden_states.size(1)
                source_len = total_len // 2
                image_hidden_states = hidden_states[:, -source_len:] * img_alpha
                hidden_states[:, -source_len:] = image_hidden_states
            else:
                hidden_states[:, :512] *= txt_alpha
                total_len = hidden_states.size(1)
                source_len = (total_len - 512) // 2
                hidden_states[:, -source_len:] *= img_alpha

            batch_size, _, _ = hidden_states.shape if not is_cross else encoder_hidden_states.shape
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

            # head 분할
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q: query = attn.norm_q(query)
            if attn.norm_k: key = attn.norm_k(key)

            # encoder projection (cross-attn 시)
            if is_cross:
                def proj(tensor, proj_fn, norm_fn):
                    out = proj_fn(tensor).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    return norm_fn(out) if norm_fn else out

                query = torch.cat([proj(encoder_hidden_states, attn.add_q_proj, attn.norm_added_q), query], dim=2)
                key = torch.cat([proj(encoder_hidden_states, attn.add_k_proj, attn.norm_added_k), key], dim=2)
                value = torch.cat([proj(encoder_hidden_states, attn.add_v_proj, None), value], dim=2)

            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

            def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                             is_causal=False, use_selective_attention=True, bos_index=0):
                B, H, L, D = query.shape
                scale = 1.0 / math.sqrt(D)
                attn_logits = torch.matmul(query, key.transpose(-2, -1)) * scale
                attn_weights = torch.softmax(attn_logits, dim=-1)

                # -------------------- 🔍 Visualize & Adjust --------------------
                def visualize_attention_maps(attn_weights, text_len, own_len, source_len, save_path):
                    B, H, L, _ = attn_weights.shape
                    pixel_start = text_len
                    pixel_own_end = pixel_start + own_len
                    pixel_source_start = pixel_own_end

                    # 1. Text attention
                    attn_text = attn_weights[:, :, pixel_start:pixel_own_end, :text_len]
                    attn_text_sum = attn_text.sum(dim=-1, keepdim=True)

                    # 2. Self attention
                    attn_self = attn_weights[:, :, pixel_start:pixel_own_end, pixel_start:pixel_own_end]
                    diag_mask = torch.eye(own_len, device=attn_self.device).unsqueeze(0).unsqueeze(0)
                    attn_self_diag = (attn_self * diag_mask).sum(dim=-1, keepdim=True)

                    # 3. Source attention
                    attn_source = attn_weights[:, :, pixel_start:pixel_own_end, pixel_source_start:]
                    diag_mask2 = torch.eye(source_len, device=attn_source.device).unsqueeze(0).unsqueeze(0)
                    attn_source_diag = (attn_source * diag_mask2).sum(dim=-1, keepdim=True)

                    # Mean & Squeeze
                    attn_text_v = attn_text_sum.mean(dim=(0, 1)).squeeze()
                    attn_self_v = attn_self_diag.mean(dim=(0, 1)).squeeze()
                    attn_source_v = attn_source_diag.mean(dim=(0, 1)).squeeze()

                    stacked = torch.stack([attn_text_v, attn_self_v, attn_source_v], dim=0)
                    stacked_norm = (stacked - stacked.min()) / (stacked.max() - stacked.min() + 1e-8)

                    grid_size = int(own_len ** 0.5)
                    stacked_2d = stacked_norm.reshape(3, grid_size, grid_size)

                    # Colormap + Resize
                    colormap = cm.get_cmap("jet")
                    img_list = []
                    for i in range(3):
                        heat = (stacked_2d[i].cpu().numpy() * 255).astype(np.uint8)
                        heat_colored = colormap(heat / 255.0)[..., :3]
                        img_colored = Image.fromarray((heat_colored * 255).astype(np.uint8)).resize((512, 512))
                        img_list.append(img_colored)

                    # Combine
                    composite = Image.new('RGB', (512 * 3, 512))
                    for i, img in enumerate(img_list):
                        composite.paste(img, (i * 512, 0))
                    #composite.save(save_path)

                # save & count
                layer_timenum = controller.layer_timenum_dict.get(layer_idx, 0)
                save_path = os.path.join(save_dir,f'attn_idx_{layer_idx}_time_{layer_timenum}.png')
                visualize_attention_maps(attn_weights, text_len=512, own_len=4096, source_len=4096,
                                         save_path=save_path)
                controller.layer_timenum_dict[layer_idx] = layer_timenum + 1
                # ------------------------------------------------------------

                attn_weights = torch.dropout(attn_weights, dropout_p, train=True)
                return torch.matmul(attn_weights, value)

            # attention 적용 후 후처리
            hidden_states = scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            if is_cross :
                encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                    [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
                )
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

                return hidden_states, encoder_hidden_states
            else:
                return hidden_states

        return forward

    # [1] image data
    image_folder = '/data/CVPR_Public/data/places365/test_samples_small_images'
    edit_file = '/data/CVPR_Public/data/places365/edit_prompt.txt'
    image_files = os.listdir(image_folder)
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image_name, _ = os.path.splitext(image_file)
        with open(edit_file, 'r') as f:
            edit_prompts = f.readlines()
        for edit_prompt in edit_prompts :
            edit_keyword = edit_prompt.split(' ')[-2]
            save_dir = os.path.join(save_folder_base, f'{image_name}_{edit_keyword}_ideal_low_{args.structure_ideal_low}_ideal_high_{args.structure_ideal_high}')
            create_folder_with_full_permissions(save_dir)
            # [1] preprocess image in same state
            input_image = Image.open(image_path).resize((1024, 1024))
            input_image.save(os.path.join(save_dir, f'original_real_image.png'))
            generator = torch.Generator().manual_seed(1983)
            img = pipe(image=input_image,
                       prompt=edit_prompt,
                       num_inference_steps=28,
                       generator=generator,
                       original=True,
                       save_dir=save_dir,
                       structure_ideal_low=args.structure_ideal_low,
                       structure_ideal_high=args.structure_ideal_high,
                       guidance_scale=2.5).images[0]
            img.save(os.path.join(save_dir, 'output.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, help='Path to the pretrained model directory')
    parser.add_argument('--save_folder', type=str, help='Directory to save results')
    parser.add_argument('--base', action='store_true', help='Use base model mode')
    parser.add_argument('--similarity_score_threds', type=float, help='Similarity score threshold')
    parser.add_argument('--structure_ideal_low', type=float, default=0.3, help='Lower bound of ideal similarity range')
    parser.add_argument('--structure_ideal_high', type=float, default=0.9, help='Upper bound of ideal similarity range')
    parser.add_argument('--use_ssim', action='store_true', help='Use base model mode')
    #

    args = parser.parse_args()
    main(args)