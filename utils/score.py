from transformers import (CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

def main() :
    print(f' step 1. call model')
    clip_id = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(clip_id)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
    image_processor = CLIPImageProcessor.from_pretrained(clip_id)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)