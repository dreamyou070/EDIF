import os
import torch
import torch
#from diffusers import FluxKontextPipeline
from model.pipeline import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image


def main() :

    print(f' step 1. Pipe')
    pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    model = pipe.transformer

    print(f' step 2. Edit with Total')
    source_folder = './prompt_analysis_edit/sources'
    edited_foldr = './prompt_analysis_edit/edited'
    positions = os.listdir(source_folder)
    for position in positions:
        if position == '5_climate' :
            source_position_folder = os.path.join(source_folder, position)
            edited_position_folder = os.path.join(edited_foldr, f'{position}_timewise_analysis')
            os.makedirs(edited_position_folder, exist_ok=True)
            images = os.listdir(source_position_folder)
            for image_idx, image in enumerate(images) :
                if image_idx == 0 :
                    input_image = Image.open(os.path.join(source_position_folder, image)).resize((1024,1024))
                    edit_prompt = "A flooded road with water covering the surface"

                    save_folder = os.path.join(edited_position_folder, f'total')
                    os.makedirs(save_folder, exist_ok=True)
                    image = pipe(image=input_image,
                                 prompt=edit_prompt,
                                 num_inference_steps=28,
                                 guidance_scale=2.5,
                                 save_latent=True,
                                 save_folder=save_folder,
                                 target_number='non',
                                 ).images[0]

                    image.save(os.path.join(save_folder, f'erase_non_0.png'))


                    """
                    # [2] prompt layerwise edit
                    class Controller(object):
                        def __init__(self):
                            self.stop = False
                            self.target_layer_number = None

                        def do_stop(self):
                            self.stop = True

                        def reset(self):
                            self.stop = False
                            self.target_layer_number = None

                    def ca_forward(attn_module, controller):  # ← 인자로 module(=attn)을 받고

                        def forward(
                                hidden_states: torch.FloatTensor,
                                encoder_hidden_states: torch.FloatTensor = None,
                                attention_mask: Optional[torch.FloatTensor] = None,
                                image_rotary_emb: Optional[torch.Tensor] = None, # text and image ids
                        ) -> torch.FloatTensor:
                            attn = attn_module  # ← 여기서 내부적으로 참조

                            if encoder_hidden_states is not None:
                                is_crossattn = True
                                encoder_hidden_states = encoder_hidden_states # * -1
                            else:
                                is_crossattn = False
                                controller.do_stop()
                            batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape # batch, text_len, dimension
                            query = attn.to_q(hidden_states)
                            key = attn.to_k(hidden_states)
                            value = attn.to_v(hidden_states)
                            inner_dim = key.shape[-1]
                            head_dim = inner_dim // attn.heads
                            # Self Attention Only 
                            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                            if attn.norm_q is not None:
                                query = attn.norm_q(query)
                            if attn.norm_k is not None:
                                key = attn.norm_k(key)

                            if encoder_hidden_states is not None:
                                # original text len = [batch, 512, dimension]
                                encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
                                encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
                                encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
                                encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                                    batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                                text_len = encoder_hidden_states_query_proj.shape[-2]
                                # batch, text_len, dimension -> batch, text_len, 24, dimension
                                # batch, 24, text_len, dimension
                                encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                                    batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                                encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                                    batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                                if attn.norm_added_q is not None:
                                    encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
                                if attn.norm_added_k is not None:
                                    encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
                                #print(f' after preprocess, encoder_hidden_states_key_proj (batch, 24, text_len(512?), dim) = {encoder_hidden_states_key_proj.shape}')
                                # ------------------------------------------------------------------------------------------
                                # Text and Image Concat on Dimension
                                # ------------------------------------------------------------------------------------------
                                #print(f' after preprocess, query (batch, 24, pixel_len, dim) = {query.shape}')
                                pixel_len = query.shape[-2]
                                query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
                                #print(f' after concat, query (batch, 24, text+pixel_len, dim) = {query.shape}')
                                # attention map : (text+pixel, text+pixe;)
                                # attention_map[:,:,pixel_len:, :text_len]

                                key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
                                value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

                            if image_rotary_emb is not None:
                                from diffusers.models.embeddings import apply_rotary_emb
                                query = apply_rotary_emb(query, image_rotary_emb)
                                key = apply_rotary_emb(key, image_rotary_emb)

                            def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                                             is_causal=False, is_crossattn=False,
                                                             text_len=0) -> torch.Tensor:
                                L, S = query.size(-2), key.size(-2)  # L: query length, S: key length
                                scale_factor = 1 / math.sqrt(query.size(-1))
                                attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

                                if is_crossattn:
                                    # key에는 [image_tokens + text_tokens]이 존재한다고 가정
                                    # text_len은 text token 개수라고 가정
                                    # text token 위치만 True로 만들고 나머지는 False

                                    
                                    # [1] 
                                    #attn_mask = torch.zeros(L, S, dtype=torch.bool, device=query.device)
                                    #attn_mask[:, :] = True
                                    #attn_mask[:, -text_len:] = False  # 마지막 text_len개에만 False
                                    
                                    # [2]
                                    attn_mask = torch.zeros(L, S, dtype=torch.bool, device=query.device)
                                    attn_mask[:, :] = False
                                    attn_mask[:, -text_len:] = True  # 마지막 text_len개에만 False
                                    # False 위치를 -inf로 만들어서 attention을 못하게 만듦
                                    attn_bias.masked_fill_(attn_mask, float("-inf"))





                                attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
                                attn_weights = attn_weights + attn_bias # bafore attention scoring_image_centric_rigid
                                attn_weights = torch.softmax(attn_weights, dim=-1)
                                attn_weights = torch.dropout(attn_weights, dropout_p, train=True)

                                return attn_weights @ value


                            hidden_states = scaled_dot_product_attention(
                                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                            )

                            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                            hidden_states = hidden_states.to(query.dtype)

                            if encoder_hidden_states is not None:
                                encoder_hidden_states, hidden_states = (
                                    hidden_states[:, : encoder_hidden_states.shape[1]],
                                    hidden_states[:, encoder_hidden_states.shape[1]:],
                                )

                                hidden_states = attn.to_out[0](hidden_states)
                                hidden_states = attn.to_out[1](hidden_states)

                                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

                                return hidden_states, encoder_hidden_states
                            else:
                                return hidden_states

                        return forward

                    targets = [i for i in range(18, -1, -1)]

                    for target_number in targets:
                        save_folder = os.path.join(edited_position_folder,f'erase_layer_number_{target_number}')
                        os.makedirs(save_folder, exist_ok=True)

                        # ------------------------------------------------------------------------------------------------
                        # Change Forward
                        # ------------------------------------------------------------------------------------------------

                        controller = Controller()
                        attention_idx = 0
                        for name, module in model.named_modules():
                            if module.__class__.__name__ == 'Attention':
                                if attention_idx == target_number:
                                    original_forward = module.forward
                                    controller.target_layer_number = target_number
                                    module.forward = ca_forward(module, controller)
                                attention_idx += 1
                        # ------------------------------------------------------------------------------------------------
                        # Inference
                        # ------------------------------------------------------------------------------------------------
                        image = pipe(image=input_image,
                                     prompt=edit_prompt,
                                     num_inference_steps=28,
                                     guidance_scale=2.5,
                                     save_latent = True,
                                     save_folder = save_folder,
                                     target_number = target_number,
                                     ).images[0]

                        image.save(os.path.join(save_folder, f'erase_{target_number}_0.png'))
                        #if not controller.stop:
                        #    edit_path = os.path.join(edited_position_folder, f'{target_number}.png')
                        controller.reset()  #
                        attention_idx = 0
                        for name, module in model.named_modules():
                            if module.__class__.__name__ == 'Attention':
                                if attention_idx == target_number:
                                    module.forward = original_forward
                                attention_idx += 1
                    """

if __name__ == '__main__':
    main()