import os
import json
import time
from datetime import datetime
from pathlib import Path
from PIL import Image

import torch
from diffusers import DiffusionPipeline
from transformers import T5EncoderModel
import torchvision.transforms.functional as TF

from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample_stage_1, sample_stage_2

###############################################
# RESUME PARAMETERS — configure these as needed
###############################################
# Which image index to resume from (0-indexed).
# Set to 0 to start from the beginning.
RESUME_FROM = 0

# If True, reuse the most recent run_* folder in outputs/
# instead of creating a new one. The prompts_log.json will
# NOT be overwritten so previous entries stay intact.
USE_LAST_FOLDER = False
###############################################

# 1. Read all files in prompts/new alphabetically and aggregate prompts
prompts_dir = Path("prompts/new")
prompt_files = sorted(prompts_dir.glob("*.json"))

prompt_pairs = []
for p_file in prompt_files:
    with open(p_file, "r", encoding="utf-8") as f:
        file_pairs = json.load(f)
        prompt_pairs.extend(file_pairs)

# Setup run directory
if USE_LAST_FOLDER:
    outputs_dir = Path("outputs")
    run_dirs = sorted(
        [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name
    )
    if not run_dirs:
        raise RuntimeError("USE_LAST_FOLDER is True but no existing run_* folders found in outputs/")
    run_dir = run_dirs[-1]
    print(f"Resuming into existing folder: {run_dir}")
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

# 2. Create the master JSON log (only if starting fresh or folder is new)
log_file_path = run_dir / "prompts_log.json"
if not USE_LAST_FOLDER or not log_file_path.exists():
    log_data = {"data": []}
    for idx, (prompt_gs, prompt_col) in enumerate(prompt_pairs):
        log_data["data"].append({
            "number": idx,
            "greyscale": prompt_gs,
            "color": prompt_col,
            "quality": ""
        })

    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)
    print(f"Master log saved to {log_file_path}")
else:
    print(f"Using existing master log at {log_file_path}")

print(f"Loaded {len(prompt_pairs)} total pairs from {len(prompt_files)} files.")
print(f"Will start generating from index {RESUME_FROM}")

def print_progress(current, total, desc):
    print(f"[{current}/{total}] {desc}", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

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
for idx, (prompt_gs, prompt_col) in enumerate(prompt_pairs):
    # Skip pairs before RESUME_FROM
    if idx < RESUME_FROM:
        continue

    print_progress(idx + 1, total, f"Generating for pair: {prompt_gs!r} | {prompt_col!r}")

    # Load text encoder and pipeline for this pair
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

    # Encode prompts for this pair
    prompts = [prompt_gs, prompt_col]
    prompt_embeds = [pipe.encode_prompt(p) for p in prompts]
    prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
    prompt_embeds = torch.cat(prompt_embeds)
    negative_prompt_embeds = torch.cat(negative_prompt_embeds)

    # Offload text encoder and pipeline to free GPU memory
    del text_encoder
    del pipe
    torch.cuda.empty_cache()

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

    def im_to_np_gray(im):
        im = (im / 2 + 0.5).clamp(0, 1)

        if im.ndim == 3:
            if im.shape[0] == 1:
                im = im[0]
            elif im.shape[0] == 3:
                im = im.mean(dim=0)

        im = im.detach().cpu().numpy()
        im = (im * 255).round().astype("uint8")
        return im

    def im_to_np(im):
        im = (im / 2 + 0.5).clamp(0, 1)
        im = im.detach().cpu().permute(1, 2, 0).numpy()
        im = (im * 255).round().astype("uint8")
        return im

    # View processing and Saving images directly to run_folder
    gs_view = views[0].view(image_1024[0])
    img_gs = im_to_np_gray(gs_view)
    
    gs_filename = f"{idx:04d}g.png"
    Image.fromarray(img_gs, mode="L").save(str(run_dir / gs_filename))
    
    col_view = views[1].view(image_1024[0])
    img_col = im_to_np(col_view)
    
    col_filename = f"{idx:04d}c.png"
    Image.fromarray(img_col).save(str(run_dir / col_filename))

print("All generations complete. Results saved to:", run_dir)
