import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from transformers import T5EncoderModel
import mediapy as mp
import torchvision.transforms.functional as TF

from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample_stage_1, sample_stage_2

def print_progress(current, total, desc):
    print(f"[{current}/{total}] {desc}", flush=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate.py <prompt_file.json>")
        sys.exit(1)
    prompt_file = sys.argv[1]
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_pairs = json.load(f)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading text encoder...")
    text_encoder = T5EncoderModel.from_pretrained(
        "DeepFloyd/IF-I-L-v1.0",
        subfolder="text_encoder",
        device_map=None,
        variant="fp16",
        torch_dtype=torch.float16,
    )
    pipe = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-L-v1.0",
        text_encoder=text_encoder,
        unet=None
    )
    pipe = pipe.to(device)

    # Encode all prompts
    print("Encoding prompts...")
    prompts_gs = [pair[0] for pair in prompt_pairs]
    prompts_col = [pair[1] for pair in prompt_pairs]
    prompt_embeds_gs = [pipe.encode_prompt(p) for p in prompts_gs]
    prompt_embeds_col = [pipe.encode_prompt(p) for p in prompts_col]
    prompt_embeds_gs, _ = zip(*prompt_embeds_gs)
    prompt_embeds_col, _ = zip(*prompt_embeds_col)
    prompt_embeds_gs = torch.cat(prompt_embeds_gs)
    prompt_embeds_col = torch.cat(prompt_embeds_col)

    del text_encoder
    del pipe
    torch.cuda.empty_cache()

    # Load image generation pipelines
    print("Loading image generation pipelines...")
    stage_1 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-L-v1.0",
        text_encoder=None,
        variant="fp16",
        torch_dtype=torch.float16,
    )
    stage_1.enable_model_cpu_offload()
    stage_1.to(device)

    stage_2 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0",
        text_encoder=None,
        variant="fp16",
        torch_dtype=torch.float16,
    )
    stage_2.enable_model_cpu_offload()
    stage_2.to(device)

    stage_3 = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=torch.float16
    )
    stage_3.enable_model_cpu_offload()
    stage_3 = stage_3.to(device)

    views = get_views(['grayscale', 'color'])

    total = len(prompt_pairs)
    for idx, (prompt_gs, prompt_col) in enumerate(prompt_pairs, 1):
        pair_dir = run_dir / str(idx)
        pair_dir.mkdir(parents=True, exist_ok=True)
        print_progress(idx, total, f"Generating for pair: {prompt_gs!r} | {prompt_col!r}")

        # Encode prompts for this pair
        prompt_embeds = torch.cat([
            prompt_embeds_gs[idx-1].unsqueeze(0),
            prompt_embeds_col[idx-1].unsqueeze(0)
        ])
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)

        # Stage 1
        image_64 = sample_stage_1(
            stage_1,
            prompt_embeds,
            negative_prompt_embeds,
            views,
            num_inference_steps=30,
            guidance_scale=10.0,
            reduction='sum',
            generator=None
        )

        # Stage 2
        image_256 = sample_stage_2(
            stage_2,
            image_64,
            prompt_embeds,
            negative_prompt_embeds,
            views,
            num_inference_steps=30,
            guidance_scale=10.0,
            reduction='sum',
            noise_level=50,
            generator=None
        )

        # Stage 3 (upsample color)
        image_1024 = stage_3(
            prompt=prompt_col,
            image=image_256,
            noise_level=0,
            output_type='pt',
            generator=None
        ).images
        image_1024 = image_1024 * 2 - 1

        # Save images
        def im_to_np(im):
            im = (im / 2 + 0.5).clamp(0, 1)
            im = im.detach().cpu().permute(1, 2, 0).numpy()
            im = (im * 255).round().astype("uint8")
            return im

        img_gs = im_to_np(views[0].view(image_1024[0]))
        img_col = im_to_np(views[1].view(image_1024[0]))
        mp.imwrite(str(pair_dir / "greyscale.png"), img_gs)
        mp.imwrite(str(pair_dir / "color.png"), img_col)

        # Save video
        from visual_anagrams.animate import animate_two_view
        pil_image = TF.to_pil_image(image_1024[0] / 2. + 0.5)
        save_video_path = pair_dir / "illusion.mp4"
        animate_two_view(
            pil_image,
            views[1],
            prompt_gs,
            prompt_col,
            save_video_path=str(save_video_path),
            hold_duration=120,
            text_fade_duration=10,
            transition_duration=45,
            im_size=img_gs.shape[0],
            frame_size=int(img_gs.shape[0] * 1.5),
        )

        # Save prompts
        with open(pair_dir / "prompts.txt", "w", encoding="utf-8") as f:
            f.write(f"Greyscale prompt: {prompt_gs}\n")
            f.write(f"Color prompt: {prompt_col}\n")

    print("All generations complete. Results saved to:", run_dir)

if __name__ == "__main__":
    main()