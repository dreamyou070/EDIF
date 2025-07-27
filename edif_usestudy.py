import torch
import os
import logging
from datetime import datetime
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",
                                                               torch_dtype="auto",
                                                               # cache_dir = '/data2/CVPR_Public/model/Qwen',
                                                               # cache_dir="/home/vmuser/.cache/huggingface",
                                                               device_map=device)
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", )  # cache_dir='/data2/CVPR_Public/model/Qwen')

    print(f' step 2. Call Flux Pipe')
    pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev",
                                               torch_dtype=torch.bfloat16, )
    pipe.to('cuda')
    flux_transformer = pipe.transformer

    print(f' (2.1) register new function')
    def onetime_inference(self, latents, image_embeds, image_latents, t, i, guidance,
                          pooled_prompt_embeds,
                          prompt_embeds, text_ids, latent_ids, callback_on_step_end,
                          callback_on_step_end_tensor_inputs, timesteps, num_warmup_steps, progress_bar,
                          height, width, output_type, start_noise,
                          logger,
                          structure_ideal_low, structure_ideal_high, ):

        #
        self._current_timestep = t
        latent_model_input = latents
        latent_model_input = torch.cat([latents, image_latents], dim=1)
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        save_time = int(timestep.round().item())
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False, )[0]
        noise_pred = noise_pred[:, : latents.size(1)]
        latents_dtype = latents.dtype

        # ---------------------------------------------------------------
        # [1] structural check using SSIM Score
        # ---------------------------------------------------------------
        def predict_x0(start_noise, noise_pred, present_latent, do_save=True, do_newname=False):
            x0_pred_latent = start_noise - noise_pred
            x0_pred_im = self._unpack_latents(x0_pred_latent, 1024, 1024, self.vae_scale_factor)
            x0_pred_im = (x0_pred_im / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            x0_pred_im = self.vae.decode(x0_pred_im, return_dict=False)[0]
            x0_pred_pil = self.image_processor.postprocess(x0_pred_im, output_type='pil')[0]
            if do_save:
                if do_newname:
                    x0_pred_pil.save(os.path.join(inter_folder, f'x0_pred_{save_time}_new.png'))
                else:
                    x0_pred_pil.save(os.path.join(inter_folder, f'x0_pred_{save_time}.png'))
            # noise_pred 이미지 저장
            noise_pred_im = self._unpack_latents(noise_pred, 1024, 1024, self.vae_scale_factor)
            noise_pred_im = (noise_pred_im / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            noise_pred_im = self.vae.decode(noise_pred_im, return_dict=False)[0]
            noise_pred_pil = self.image_processor.postprocess(noise_pred_im, output_type='pil')[0]

            present_latent_im = self._unpack_latents(present_latent, 1024, 1024, self.vae_scale_factor)
            present_latent_im = (present_latent_im / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            present_latent_im = self.vae.decode(present_latent_im, return_dict=False)[0]
            present_latent_pil = self.image_processor.postprocess(present_latent_im, output_type='pil')[0]
            if do_save:
                if do_newname:
                    present_latent_pil.save(os.path.join(inter_folder, f'present_latent_{save_time}_new.png'))
                else:
                    present_latent_pil.save(os.path.join(inter_folder, f'present_latent_{save_time}.png'))
            return x0_pred_latent, x0_pred_pil, present_latent_pil

        # ------------------------------------------------------------------------------------
        def set_compare_tensor(inference_step, x0_pred, x0_pred_pil, present_latent, present_latent_pil):
            if inference_step > -1:
                if save_time > 300:
                    compare_tensor = x0_pred
                    compare_pil = x0_pred_pil
                else:
                    compare_tensor = present_latent
                    compare_pil = present_latent_pil
            return compare_tensor, compare_pil

        # -------------------------------------------------------------------------------------
        # control image condition : 구조 유사도 너무 낮은 경우 → 이미지 latent 강화, 텍스트 latent 약화
        from diffusers.training_utils import compute_density_for_timestep_sampling
        def get_sigmas(i, n_dim=4, dtype=torch.float32):
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            sigmas = torch.tensor(sigmas, device=image_latents.device, dtype=dtype)
            sigma = sigmas[i].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

        sigmas = get_sigmas(i, n_dim=image_latents.ndim, dtype=image_latents.dtype)

        x0_pred_latent, x0_pred_pil, present_latent_pil = predict_x0(start_noise, noise_pred, latents, do_save=True)
        # ------------------------------------------------------
        x0_pred_latent = latents - sigmas * start_noise
        # ------------------------------------------------------
        compare_tensor, compare_pil = set_compare_tensor(timestep, x0_pred_latent, x0_pred_pil, latents,
                                                         present_latent_pil)
        compare_pil.save(os.path.join(inter_folder, f'comparison_{save_time}.png'))

        def get_structural_similarity(compare_pil, input_image):
            image1 = np.array(compare_pil.resize((1024, 1024)).convert('L'))
            image2 = np.array(input_image.resize((1024, 1024)).convert('L'))
            structure_similarity = image_comparison(image1, image2)
            return structure_similarity

        structure_similarity = get_structural_similarity(compare_pil, input_image)
        logger.info(f"[{save_time}] structure_similarity: {structure_similarity}")
        ideal_structure_similar_range = (structure_ideal_low, structure_ideal_high)

        if structure_similarity < structure_ideal_low and i < 1:
            logger.info(f' ******************* Use New Noise Pred!')
            # noise_pred = start_noise - image_latents
            latents_new = (1.0 - sigmas) * image_latents + sigmas * start_noise
            latents = latents_new

            latent_model_input_new = torch.cat([latents, image_latents], dim=1)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            save_time = int(timestep.round().item())
            noise_pred = self.transformer(
                hidden_states=latent_model_input_new,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False, )[0]
            noise_pred = noise_pred[:, : latents.size(1)]
            ############################################################################################################
            x0_pred_latent, x0_pred_pil, present_latent_pil = predict_x0(start_noise, noise_pred, latents,
                                                                         do_save=True, do_newname=True)
            # ------------------------------------------------------
            x0_pred_latent = latents - sigmas * start_noise
            # ------------------------------------------------------
            compare_tensor, compare_pil = set_compare_tensor(timestep, x0_pred_latent, x0_pred_pil, latents,
                                                             present_latent_pil)
            compare_pil.save(os.path.join(inter_folder, f'comparison_{save_time}.png'))
            ############################################################################################################
            structure_similarity = get_structural_similarity(compare_pil, input_image)
            logger.info(f' [{save_time}] new structure_similarity: {structure_similarity}')

        # -------------------------------------------------------------------------------------
        # Compare Again
        # -------------------------------------------------------------------------------------
        logger.info(
            f'[⚠] structure_similarity {structure_similarity} is under ideal ({ideal_structure_similar_range[0]}) → image ↑')
        if i < 3:
            if structure_similarity > ideal_structure_similar_range[1]:  # too similar
                delta_img = 0  # .1
                logger.info(
                    f'[⚠] structure_similarity {structure_similarity} is over ideal ({ideal_structure_similar_range[1]}) → image ↓')

                for k in controller.img_alpha_dict:
                    old_img_alpha = controller.img_alpha_dict[k]
                    if k in QIB:
                        new_img_alpha = max(0.8, old_img_alpha - delta_img * 0.05)
                        action = "유지 (QIB)"
                    elif k in SIB:
                        new_img_alpha = max(0.6, old_img_alpha - delta_img)
                        action = f"↓ {delta_img:.2f} (SIB)"
                    else:
                        new_img_alpha = max(0.6, old_img_alpha - delta_img * 0.5)
                        action = f"↓ {delta_img * 0.5:.2f} (Other)"

                    controller.img_alpha_dict[k] = new_img_alpha

                    if k == 0:
                        logger.info(
                            f'  [↓] img_alpha[{k}]: {old_img_alpha:.2f} → {new_img_alpha:.2f} | {action}'
                        )

            elif structure_similarity < ideal_structure_similar_range[0]:
                delta_img = 0  # .1
                logger.info(
                    f'[⚠] structure_similarity {structure_similarity} is too low ideal ({ideal_structure_similar_range[0]}) → image ↑')

                for k in controller.img_alpha_dict:
                    old_img_alpha = controller.img_alpha_dict[k]
                    if k in QIB:
                        new_img_alpha = max(1.2, old_img_alpha + delta_img * 0.05)
                        action = "유지 또는 미소 증가 (QIB)"
                    elif k in SIB:
                        new_img_alpha = max(2.0, old_img_alpha + delta_img)
                        action = f"↑ {delta_img:.2f} (SIB)"
                    else:
                        new_img_alpha = max(2.0, old_img_alpha + delta_img * 0.5)
                        action = f"↑ {delta_img * 0.5:.2f} (Other)"

                    controller.img_alpha_dict[k] = new_img_alpha

                    if k == 0:
                        logger.info(
                            f'  [↑] img_alpha[{k}]: {old_img_alpha:.2f} → {new_img_alpha:.2f} | {action}')

        # ------------------------------------------------------------------------------------------------------------------------------
        # [2] Text Scoring  # 0.6 ~ 2.0
        # ------------------------------------------------------------------------------------------------------------------------------
        def check_text_editing():
            edited_img_dir = os.path.join(inter_folder, f'comparison_{save_time}.png')
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": edited_img_dir},
                        {"type": "text", "text": f"이 이미지에 {user_edit_keyword} 가 잘 표현되었니?"
                                                 f"잘 표현되었다면 Yes 로 답하고 아니면 No 로 답해줘. 다른 답은 말고, Yes 와 No 로만 답해줌"},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                conversation,
                video_fps=1,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)
            output_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)[0]
            return output_text.strip().lower()

        # 최대 3회까지 재시도
        max_retry = 3
        txt_decision = None
        for attempt in range(max_retry):
            txt_decision = check_text_editing()
            if txt_decision.lower() in ['yes', 'no']:
                break
            else:
                logger.info(f"[{attempt + 1}/{max_retry}] Unexpected response: '{txt_decision}' → retrying...")

        logger.info(f'VLM txt_decision = {txt_decision}')

        if txt_decision.lower() not in ['yes', 'no']:
            logger.info(f"[✘] Failed to get valid decision after {max_retry} attempts: '{txt_decision}'")
            # 여기서 fallback 로직이나 예외처리 가능

        delta_txt = 0.2  # 조정 범위
        if txt_decision.strip().lower() == 'no':
            logger.info(f"[✘] {user_edit_keyword} 가 잘 표현되지 않음 → 텍스트 알파 ↑")
            for k in controller.txt_alpha_dict:
                old_txt_alpha = controller.txt_alpha_dict[k]
                if k in QTB:
                    new_txt_alpha = min(2.0, old_txt_alpha + delta_txt * 0.5)
                    action = f"↑ {delta_txt:.2f} (QTB)"
                elif k in ETB:
                    new_txt_alpha = min(3.0, old_txt_alpha + delta_txt * 1.0)
                    action = f"↑ {delta_txt:.2f} (ETB)"
                else:
                    new_txt_alpha = min(3.0, old_txt_alpha + delta_txt * 1.0)
                    action = f"↑ {delta_txt * 0.5:.2f} (Other)"
                controller.txt_alpha_dict[k] = new_txt_alpha
                if k == 0:
                    logger.info(f"  [↑] txt_alpha[{k}]: {old_txt_alpha:.2f} → {new_txt_alpha:.2f} | {action}")

            # for k in controller.img_alpha_dict:
            #    delta_img = 0.3
            #    old_img_alpha = controller.img_alpha_dict[k]
            #    if k in SIB:
            #        new_img_alpha = max(0.2, old_img_alpha - delta_img)
            #        action = f"↓ {delta_img:.2f} (SIB)"
            #        controller.img_alpha_dict[k] = new_img_alpha
            #        print(f"  [↑] img_alpha[{k}]: also decrease {old_img_alpha:.2f} → {new_img_alpha:.2f} ")



        elif txt_decision.strip().lower() == 'yes':
            # txt 가 잘 표현됨
            # if i < 3 and structure_similarity > structure_ideal_low : # use good
            #    print(f'change original latents !!')
            #    image_latents = compare_tensor
            # txt 를 1로 유지해라
            # for k in controller.txt_alpha_dict:
            #    old_txt_alpha = controller.txt_alpha_dict[k]
            #    new_txt_alpha = 1
            #    controller.txt_alpha_dict[k] = 1
            logger.info(f"[✓] '{user_edit_keyword}' 가 잘 표현됨 → txt 그대로 유지 + start latent change")
        else:
            logger.info(f"[!] 텍스트 인식 실패 또는 이상 응답: {txt_decision.strip()} → check_text_editing() 재시도 또는 예외 처리 필요")
        img_alpha_dict = controller.img_alpha_dict
        txt_alpha_dict = controller.txt_alpha_dict
        # 가장 첫 번째 key 기준으로 정렬하여 값 출력
        first_img_key = sorted(img_alpha_dict.keys())[0]
        first_txt_key = sorted(txt_alpha_dict.keys())[0]
        logger.info(f'# 첫번째 img alpha dict: key = {first_img_key}, value = {img_alpha_dict[first_img_key]}')
        logger.info(f'# 첫번째 txt alpha dict: key = {first_txt_key}, value = {txt_alpha_dict[first_txt_key]}')
        latents_dtype = latents.dtype

        # -----------------------------------------------------------------
        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            if torch.backends.mps.is_available():
                latents = latents.to(latents_dtype)
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            if progress_bar is not None:
                progress_bar.update()
        logger.info(f' ============================================================================ ')
        return latents, image_latents

    pipe.onetime_inference = MethodType(onetime_inference, pipe)
    print(f' step 3. save folder')
    save_folder_base = '/data2/CVPR_Public/Kontext_EDIF_userstudy'
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
                    # composite.save(save_path)

                # save & count
                layer_timenum = controller.layer_timenum_dict.get(layer_idx, 0)
                save_path = os.path.join(save_dir, f'attn_idx_{layer_idx}_time_{layer_timenum}.png')
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

            if is_cross:
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
    # 13 번
    # 352 번
    image_folder = '../user_study'
    edit_file = '../user_study/demo.json'
    import json
    with open(edit_file, 'r', encoding='utf-8') as f:
        contents = json.load(f)

    for data in contents:
        file = data['filename']
        image_name, _ = os.path.splitext(file)
        edit_prompt = data["scene_change"][0]
        edit_keyword = data["scene_change"][1]
        res = 1024
        save_dir = os.path.join(save_folder_base,
                                f'{image_name}_{edit_keyword}_ideal_low_{args.structure_ideal_low}'
                                f'_ideal_high_{args.structure_ideal_high}')
        # if not os.path.exists(save_dir):
        create_folder_with_full_permissions(save_dir)
        inter_folder = os.path.join(save_dir, 'inter')
        create_folder_with_full_permissions(inter_folder)
        # [1] preprocess image in same state
        image_path = os.path.join(image_folder, file)
        input_image = Image.open(image_path).resize((res, res))
        input_image.save(os.path.join(save_dir, f'original_real_image.png'))
        user_edit_keyword = edit_prompt

        print(f' step 3. Set Controller')
        controller = Controller()
        attention_idx = 0
        for name, module in flux_transformer.named_modules():
            if module.__class__.__name__ == 'FluxAttention' or module.__class__.__name__ == 'Attention':
                txt_alpha = 1
                img_alpha = 1
                controller.set_alpha(attention_idx, txt_alpha, img_alpha)
                module.forward = ca_forward(module, attention_idx, controller)
                attention_idx += 1
        generator = torch.Generator().manual_seed(1983)
        print(f' ************** {save_dir} *********************** ')

        def setup_logger(log_path):
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            logger = logging.getLogger(os.path.basename(log_path))  # 이름 중복 방지
            logger.setLevel(logging.INFO)

            # if not logger.handlers:
            file_handler = logging.FileHandler(log_path, mode='a')  # 또는 'w'
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            # 콘솔 핸들러 (→ 화면에 출력)
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            return logger

        # 예시 변수들
        log_dir = os.path.join(save_dir, f'record_log_{image_name}_{edit_keyword}.log')
        logger = setup_logger(log_dir)
        logger.info("이것은 파일에 저장될 로그입니다.")
        num_inference_steps = 28
        img = pipe(image=input_image,
                   prompt=edit_prompt,
                   num_inference_steps=num_inference_steps,
                   generator=generator,
                   original=False,
                   save_dir=save_dir,
                   structure_ideal_low=args.structure_ideal_low,
                   structure_ideal_high=args.structure_ideal_high,
                   logger=logger,
                   guidance_scale=2.5).images[0]
        img.save(os.path.join(save_dir, 'output.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, help='Path to the pretrained model directory')
    parser.add_argument('--save_folder', type=str, help='Directory to save results')
    parser.add_argument('--base', action='store_true', help='Use base model mode')
    parser.add_argument('--similarity_score_threds', type=float, help='Similarity score threshold')
    parser.add_argument('--structure_ideal_low', type=float, default=0.3,
                        help='Lower bound of ideal similarity range')
    parser.add_argument('--structure_ideal_high', type=float, default=0.9, help='Upper bound of ideal similarity range')
    parser.add_argument('--use_ssim', action='store_true', help='Use base model mode')
    #

    args = parser.parse_args()
    main(args)