"""
================================================================================
MASTER EVALUATION PIPELINE — master_eval.py
================================================================================

PURPOSE:
    Unified evaluation of 5 Vision-Language Models on color-hybrid illusion images.
    Tests whether VLMs rely more on chromatic or luminance cues when they conflict.

MODELS (toggle each on/off via booleans at top):
    1. openai/clip-vit-base-patch32        — Classification only  (zero-shot, softmax)
    2. google/siglip-base-patch16-224      — Classification only  (zero-shot, sigmoid)
    3. Salesforce/blip2-opt-2.7b           — Generation + Classification
    4. llava-hf/llava-1.5-7b-hf           — Generation + Classification
    5. llava-hf/llava-v1.6-mistral-7b-hf  — Generation + Classification

EVALUATION MODES:
    Classification — Forced-choice between the two entity labels (grey_object vs
                     color_object). The model picks whichever label scores higher.
                     Metric: binary accuracy (correct / total).
    Generation     — Open-ended VQA ("What is the main object?"). The free-text
                     answer is scored against the ground-truth label via cosine
                     similarity using SentenceTransformer (all-MiniLM-L6-v2).
                     Metric: semantic similarity score.

DATA:
    Reads 1.json which contains the dataset of image pairs with fields:
        number, greyscale, color, quality, grey_object, color_object
    Images are at data/<number>g.png (greyscale) and data/<number>c.png (color).

OUTPUT STRUCTURE:
    master_results/<timestamp>/
    ├── raw_data/
    │   ├── clip_classification.csv
    │   ├── siglip_classification.csv
    │   ├── blip2_classification.csv
    │   ├── blip2_generation.csv
    │   ├── llava15_classification.csv
    │   ├── llava15_generation.csv
    │   ├── llava16_classification.csv
    │   ├── llava16_generation.csv
    │   ├── align_classification.csv
    │   ├── gpt4o_mini_classification.csv
    │   ├── gpt4o_mini_generation.csv
    │   ├── gpt55_classification.csv
    │   ├── gpt55_generation.csv
    │   ├── smolvlm_classification.csv
    │   ├── smolvlm_generation.csv
    │   ├── qwen2vl_classification.csv
    │   ├── qwen2vl_generation.csv
    │   ├── moondream2_classification.csv
    │   └── moondream2_generation.csv
    ├── charts/
    │   ├── <model>_accuracy_by_type.png          — per-model bar chart
    │   ├── <model>_confusion_matrix.png          — per-model confusion matrix
    │   ├── <model>_similarity_by_type.png         — per-model gen similarity
    │   ├── cross_model_classification_accuracy.png — grouped bar across models
    │   ├── cross_model_generation_similarity.png   — grouped bar across gen models
    │   ├── cross_model_accuracy_by_quality.png     — accuracy vs quality tier
    │   └── cross_model_grey_vs_color_delta.png     — grey-color accuracy gap
    └── consistent_performers_classification.csv  — image pairs all models agree on
        consistent_performers_generation.csv       — image pairs all gen models agree on
================================================================================
"""

import os
import json
import time
import gc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import seaborn as sns
import numpy as np
from PIL import Image
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TOGGLES — set to True/False to enable/disable each model
# ═══════════════════════════════════════════════════════════════════════════════
RUN_CLIP       = True   # openai/clip-vit-base-patch32        (classification)
RUN_SIGLIP     = True   # google/siglip-base-patch16-224      (classification)
RUN_ALIGN      = True   # kakaobrain/align-base               (classification)
RUN_BLIP2      = True   # Salesforce/blip2-opt-2.7b           (gen + classification)
RUN_LLAVA_15   = True   # llava-hf/llava-1.5-7b-hf           (gen + classification)
RUN_LLAVA_16   = True   # llava-hf/llava-v1.6-mistral-7b-hf  (gen + classification)
RUN_GPT4O_MINI = True   # OpenAI gpt-4o-mini API             (gen + classification)
RUN_GPT55      = True   # OpenAI gpt-5.5 API                 (gen + classification)
RUN_SMOLVLM    = True   # HuggingFaceTB/SmolVLM-Instruct      (gen + classification)
RUN_QWEN2VL    = True   # Qwen/Qwen2-VL-7B-Instruct          (gen + classification)
RUN_MOONDREAM2 = True   # vikhyatk/moondream2                 (gen + classification)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA / PATH CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
DATA_JSON   = "1.json"
IMAGE_DIR   = "data"
SIM_MODEL   = "all-MiniLM-L6-v2"  # SentenceTransformer for generation eval

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_data():
    with open(DATA_JSON, "r") as f:
        log_data = json.load(f)
    items = log_data.get("data", [])
    valid = [i for i in items if i.get("grey_object") and i.get("color_object")]
    print(f"Loaded {len(valid)} valid items from {DATA_JSON}")
    return valid

def get_image_paths(item):
    num = f"{item['number']:04d}"
    return os.path.join(IMAGE_DIR, f"{num}g.png"), os.path.join(IMAGE_DIR, f"{num}c.png")

def free_model(*models):
    """Aggressively free GPU memory between model runs."""
    # Synchronize to make sure all GPU ops are done
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Move each model to CPU before deleting (releases GPU tensors)
    for m in models:
        if hasattr(m, 'cpu'):
            try:
                m.cpu()
            except Exception:
                pass
        del m
    # Force garbage collection and clear CUDA cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print(f"  [VRAM] Freed model memory. ", end="")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU mem: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print("(no CUDA device)")

# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION EVALUATORS
# ═══════════════════════════════════════════════════════════════════════════════

def eval_clip_classification(items, device):
    """CLIP ViT-B/32 — softmax over [grey_label, color_label]."""
    from transformers import CLIPProcessor, CLIPModel
    model_name = "openai/clip-vit-base-patch32"
    print(f"\n{'='*60}\n  CLIP Classification: {model_name}\n{'='*60}")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    results = []
    total = len(items)
    for i, item in enumerate(items, 1):
        grey_path, color_path = get_image_paths(item)
        grey_obj, color_obj = item["grey_object"], item["color_object"]
        labels = [f"a photo of a {grey_obj}", f"a photo of a {color_obj}"]

        for img_path, true_obj, img_type in [
            (grey_path, grey_obj, "grey"),
            (color_path, color_obj, "color"),
        ]:
            if not os.path.exists(img_path):
                print(f"  Warning: {img_path} not found")
                continue
            image = Image.open(img_path).convert("RGB")
            inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            pred_is_grey = probs[0] > probs[1]
            pred_obj = grey_obj if pred_is_grey else color_obj
            results.append({
                "model": "CLIP",
                "image_name": os.path.basename(img_path),
                "image_type": img_type,
                "true_object": true_obj,
                "pred_object": pred_obj,
                "prob_grey_object": float(probs[0]),
                "prob_color_object": float(probs[1]),
                "correct": pred_obj == true_obj,
                "quality": item.get("quality", ""),
            })
        if i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] {i/total*100:.1f}%")

    free_model(model, processor)
    return pd.DataFrame(results)


def eval_siglip_classification(items, device):
    """SigLIP — sigmoid over [grey_label, color_label]."""
    from transformers import AutoProcessor, AutoModel
    model_name = "google/siglip-base-patch16-224"
    print(f"\n{'='*60}\n  SigLIP Classification: {model_name}\n{'='*60}")
    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)

    results = []
    total = len(items)
    for i, item in enumerate(items, 1):
        grey_path, color_path = get_image_paths(item)
        grey_obj, color_obj = item["grey_object"], item["color_object"]
        labels = [f"a photo of a {grey_obj}", f"a photo of a {color_obj}"]

        for img_path, true_obj, img_type in [
            (grey_path, grey_obj, "grey"),
            (color_path, color_obj, "color"),
        ]:
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            inputs = processor(text=labels, images=image, padding="max_length", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits_per_image).cpu().numpy()[0]
            pred_is_grey = probs[0] > probs[1]
            pred_obj = grey_obj if pred_is_grey else color_obj
            results.append({
                "model": "SigLIP",
                "image_name": os.path.basename(img_path),
                "image_type": img_type,
                "true_object": true_obj,
                "pred_object": pred_obj,
                "prob_grey_object": float(probs[0]),
                "prob_color_object": float(probs[1]),
                "correct": pred_obj == true_obj,
                "quality": item.get("quality", ""),
            })
        if i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] {i/total*100:.1f}%")

    free_model(model, processor)
    return pd.DataFrame(results)


def eval_align_classification(items, device):
    """ALIGN — dot product over [grey_label, color_label]."""
    from transformers import AlignProcessor, AlignModel
    model_name = "kakaobrain/align-base"
    print(f"\n{'='*60}\n  ALIGN Classification: {model_name}\n{'='*60}")
    model = AlignModel.from_pretrained(model_name).to(device)
    processor = AlignProcessor.from_pretrained(model_name)

    results = []
    total = len(items)
    for i, item in enumerate(items, 1):
        grey_path, color_path = get_image_paths(item)
        grey_obj, color_obj = item["grey_object"], item["color_object"]
        labels = [f"a photo of a {grey_obj}", f"a photo of a {color_obj}"]

        for img_path, true_obj, img_type in [
            (grey_path, grey_obj, "grey"),
            (color_path, color_obj, "color"),
        ]:
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            pred_is_grey = probs[0] > probs[1]
            pred_obj = grey_obj if pred_is_grey else color_obj
            results.append({
                "model": "ALIGN",
                "image_name": os.path.basename(img_path),
                "image_type": img_type,
                "true_object": true_obj,
                "pred_object": pred_obj,
                "prob_grey_object": float(probs[0]),
                "prob_color_object": float(probs[1]),
                "correct": pred_obj == true_obj,
                "quality": item.get("quality", ""),
            })
        if i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] {i/total*100:.1f}%")

    free_model(model, processor)
    return pd.DataFrame(results)


def eval_blip2_classification(items, device, processor, model, sim_model):
    """BLIP-2 forced-choice: ask 'Is this a {A} or a {B}?' and match answer."""
    from sentence_transformers import util
    print(f"\n{'='*60}\n  BLIP-2 Classification (forced-choice VQA)\n{'='*60}")

    results = []
    total = len(items)
    for i, item in enumerate(items, 1):
        grey_path, color_path = get_image_paths(item)
        grey_obj, color_obj = item["grey_object"], item["color_object"]

        for img_path, true_obj, img_type in [
            (grey_path, grey_obj, "grey"),
            (color_path, color_obj, "color"),
        ]:
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            question = f"Is this a {grey_obj} or a {color_obj}?"
            inputs = processor(images=image, text=question, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=20)
            answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip().lower()

            # Score answer against both candidates
            emb_answer = sim_model.encode(answer, convert_to_tensor=True)
            emb_grey = sim_model.encode(grey_obj, convert_to_tensor=True)
            emb_color = sim_model.encode(color_obj, convert_to_tensor=True)
            sim_grey = util.pytorch_cos_sim(emb_answer, emb_grey).item()
            sim_color = util.pytorch_cos_sim(emb_answer, emb_color).item()

            pred_obj = grey_obj if sim_grey > sim_color else color_obj
            results.append({
                "model": "BLIP-2",
                "image_name": os.path.basename(img_path),
                "image_type": img_type,
                "true_object": true_obj,
                "pred_object": pred_obj,
                "vlm_answer": answer,
                "sim_to_grey": float(sim_grey),
                "sim_to_color": float(sim_color),
                "correct": pred_obj == true_obj,
                "quality": item.get("quality", ""),
            })
        if i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] {i/total*100:.1f}%")

    return pd.DataFrame(results)


def eval_blip2_generation(items, device, processor, model, sim_model):
    """BLIP-2 open-ended: 'What is the main object in this image?'"""
    from sentence_transformers import util
    print(f"\n{'='*60}\n  BLIP-2 Generation (open-ended VQA)\n{'='*60}")

    results = []
    total = len(items)
    for i, item in enumerate(items, 1):
        grey_path, color_path = get_image_paths(item)
        grey_obj, color_obj = item["grey_object"], item["color_object"]

        for img_path, true_obj, img_type in [
            (grey_path, grey_obj, "grey"),
            (color_path, color_obj, "color"),
        ]:
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            question = "What is the main object in this image?"
            inputs = processor(images=image, text=question, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=20)
            vlm_guess = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip().lower()

            emb_guess = sim_model.encode(vlm_guess, convert_to_tensor=True)
            emb_true = sim_model.encode(true_obj, convert_to_tensor=True)
            cos_sim = util.pytorch_cos_sim(emb_guess, emb_true).item()

            results.append({
                "model": "BLIP-2",
                "image_name": os.path.basename(img_path),
                "image_type": img_type,
                "true_object": true_obj,
                "vlm_guess": vlm_guess,
                "similarity_score": float(cos_sim),
                "quality": item.get("quality", ""),
            })
        if i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] {i/total*100:.1f}%")

    return pd.DataFrame(results)


def run_blip2(items, device):
    """Load BLIP-2 once, run both classification and generation."""
    from transformers import AutoProcessor, Blip2ForConditionalGeneration
    from sentence_transformers import SentenceTransformer
    model_name = "Salesforce/blip2-opt-2.7b"
    print(f"\n  Loading BLIP-2: {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    sim_model = SentenceTransformer(SIM_MODEL, device=device)

    df_cls = eval_blip2_classification(items, device, processor, model, sim_model)
    df_gen = eval_blip2_generation(items, device, processor, model, sim_model)

    free_model(model, processor, sim_model)
    return df_cls, df_gen


# ── LLaVA helpers ─────────────────────────────────────────────────────────────

def _llava_classify(items, device, processor, model, sim_model, prompt_template, model_label):
    """LLaVA forced-choice classification via VQA."""
    from sentence_transformers import util
    print(f"\n{'='*60}\n  {model_label} Classification (forced-choice VQA)\n{'='*60}")

    results = []
    total = len(items)
    for i, item in enumerate(items, 1):
        grey_path, color_path = get_image_paths(item)
        grey_obj, color_obj = item["grey_object"], item["color_object"]

        for img_path, true_obj, img_type in [
            (grey_path, grey_obj, "grey"),
            (color_path, color_obj, "color"),
        ]:
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            prompt = prompt_template.format(grey_obj=grey_obj, color_obj=color_obj)
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=15, use_cache=True)
            generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
            answer = processor.decode(generated_ids, skip_special_tokens=True).strip().lower()

            emb_answer = sim_model.encode(answer, convert_to_tensor=True)
            emb_grey = sim_model.encode(grey_obj, convert_to_tensor=True)
            emb_color = sim_model.encode(color_obj, convert_to_tensor=True)
            sim_grey = util.pytorch_cos_sim(emb_answer, emb_grey).item()
            sim_color = util.pytorch_cos_sim(emb_answer, emb_color).item()

            pred_obj = grey_obj if sim_grey > sim_color else color_obj
            results.append({
                "model": model_label,
                "image_name": os.path.basename(img_path),
                "image_type": img_type,
                "true_object": true_obj,
                "pred_object": pred_obj,
                "vlm_answer": answer,
                "sim_to_grey": float(sim_grey),
                "sim_to_color": float(sim_color),
                "correct": pred_obj == true_obj,
                "quality": item.get("quality", ""),
            })
        if i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] {i/total*100:.1f}%")

    return pd.DataFrame(results)


def _llava_generate(items, device, processor, model, sim_model, prompt_template, model_label):
    """LLaVA open-ended generation."""
    from sentence_transformers import util
    print(f"\n{'='*60}\n  {model_label} Generation (open-ended VQA)\n{'='*60}")

    results = []
    total = len(items)
    for i, item in enumerate(items, 1):
        grey_path, color_path = get_image_paths(item)
        grey_obj, color_obj = item["grey_object"], item["color_object"]

        for img_path, true_obj, img_type in [
            (grey_path, grey_obj, "grey"),
            (color_path, color_obj, "color"),
        ]:
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            inputs = processor(text=prompt_template, images=image, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=15, use_cache=True)
            generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
            vlm_guess = processor.decode(generated_ids, skip_special_tokens=True).strip().lower()

            emb_guess = sim_model.encode(vlm_guess, convert_to_tensor=True)
            emb_true = sim_model.encode(true_obj, convert_to_tensor=True)
            cos_sim = util.pytorch_cos_sim(emb_guess, emb_true).item()

            results.append({
                "model": model_label,
                "image_name": os.path.basename(img_path),
                "image_type": img_type,
                "true_object": true_obj,
                "vlm_guess": vlm_guess,
                "similarity_score": float(cos_sim),
                "quality": item.get("quality", ""),
            })
        if i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] {i/total*100:.1f}%")

    return pd.DataFrame(results)


def run_llava15(items, device):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from sentence_transformers import SentenceTransformer
    model_name = "llava-hf/llava-1.5-7b-hf"
    label = "LLaVA-1.5"
    print(f"\n  Loading {label}: {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    sim_model = SentenceTransformer(SIM_MODEL, device=device)

    cls_prompt = "USER: <image>\nIs this a {grey_obj} or a {color_obj}? Reply with just the object name.\nASSISTANT:"
    gen_prompt = "USER: <image>\nWhat is the main object in this image? Reply with just one specific noun.\nASSISTANT:"

    df_cls = _llava_classify(items, device, processor, model, sim_model, cls_prompt, label)
    df_gen = _llava_generate(items, device, processor, model, sim_model, gen_prompt, label)

    free_model(model, processor, sim_model)
    return df_cls, df_gen


def run_llava16(items, device):
    from transformers import AutoProcessor, LlavaNextForConditionalGeneration
    from sentence_transformers import SentenceTransformer
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    label = "LLaVA-1.6"
    print(f"\n  Loading {label}: {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    sim_model = SentenceTransformer(SIM_MODEL, device=device)

    cls_prompt = "[INST] <image>\nIs this a {grey_obj} or a {color_obj}? Reply with just the object name. [/INST]"
    gen_prompt = "[INST] <image>\nWhat is the main object in this image? Reply with just one specific noun. [/INST]"

    df_cls = _llava_classify(items, device, processor, model, sim_model, cls_prompt, label)
    df_gen = _llava_generate(items, device, processor, model, sim_model, gen_prompt, label)

    free_model(model, processor, sim_model)
    return df_cls, df_gen


# ═══════════════════════════════════════════════════════════════════════════════
# CHART GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

# Nice color palette
COLORS = {
    "CLIP":        "#4E79A7",
    "SigLIP":      "#F28E2B",
    "ALIGN":       "#F1CE63",
    "BLIP-2":      "#E15759",
    "LLaVA-1.5":   "#76B7B2",
    "LLaVA-1.6":   "#59A14F",
    "GPT-4o-mini": "#EDC948",
    "GPT-5.5":     "#E5B944",
    "SmolVLM":     "#B07AA1",
    "Qwen2-VL":    "#FF9DA7",
    "Moondream2":   "#9C755F",
}

def _style_setup():
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["font.family"] = "sans-serif"


def _add_bar_labels(ax, fmt=".1%", fontsize=10):
    """Annotate each bar with its value."""
    for p in ax.patches:
        h = p.get_height()
        if pd.notna(h) and h > 0:
            ax.annotate(
                f"{h:{fmt}}" if "%" in fmt else f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
            )


def _add_bar_labels_decimal(ax, fontsize=10):
    """Annotate bars with decimal accuracy values (e.g. 0.630)."""
    for p in ax.patches:
        h = p.get_height()
        if pd.notna(h) and h > 0:
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
            )


def _add_bar_labels_float(ax, fontsize=10):
    """Annotate bars with float values."""
    for p in ax.patches:
        h = p.get_height()
        if pd.notna(h) and h > 0:
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
            )


def generate_per_model_charts(df_cls, df_gen, model_label, charts_dir):
    """Generate per-model accuracy / similarity charts."""
    safe = model_label.replace(" ", "_").replace(".", "")

    # ── Classification charts ─────────────────────────────────────────────
    if df_cls is not None and len(df_cls) > 0:
        # Accuracy by image type
        fig, ax = plt.subplots(figsize=(7, 5))
        acc = df_cls.groupby("image_type")["correct"].mean()
        bars = ax.bar(acc.index, acc.values, color=[COLORS.get(model_label, "#999")] * len(acc), edgecolor="white", width=0.5)
        _add_bar_labels_decimal(ax)
        ax.set_title(f"Classification Accuracy by Image Type — {model_label}", fontweight="bold")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.15)
        plt.savefig(os.path.join(charts_dir, f"{safe}_cls_accuracy_by_type.png"))
        plt.close()

        # Confusion matrix (grey vs color prediction)
        df_cls_copy = df_cls.copy()
        if "prob_grey_object" in df_cls_copy.columns:
            df_cls_copy["pred_type"] = df_cls_copy.apply(
                lambda r: "grey" if r["prob_grey_object"] > r["prob_color_object"] else "color", axis=1
            )
        elif "sim_to_grey" in df_cls_copy.columns:
            df_cls_copy["pred_type"] = df_cls_copy.apply(
                lambda r: "grey" if r["sim_to_grey"] > r["sim_to_color"] else "color", axis=1
            )
        else:
            df_cls_copy["pred_type"] = "unknown"

        if "pred_type" in df_cls_copy.columns and df_cls_copy["pred_type"].nunique() > 1:
            cm = pd.crosstab(df_cls_copy["image_type"], df_cls_copy["pred_type"],
                             rownames=["True"], colnames=["Predicted"])
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Confusion Matrix — {model_label}", fontweight="bold")
            plt.savefig(os.path.join(charts_dir, f"{safe}_cls_confusion_matrix.png"))
            plt.close()

        # Accuracy by quality tier
        if "quality" in df_cls.columns and df_cls["quality"].str.strip().ne("").any():
            fig, ax = plt.subplots(figsize=(7, 5))
            quality_order = ["L", "M", "H"]
            acc_q = df_cls.groupby("quality")["correct"].mean().reindex(quality_order).dropna()
            if len(acc_q) > 0:
                bars = ax.bar(acc_q.index, acc_q.values, color=COLORS.get(model_label, "#999"), edgecolor="white", width=0.5)
                _add_bar_labels_decimal(ax)
                ax.set_title(f"Classification Accuracy by Quality — {model_label}", fontweight="bold")
                ax.set_ylabel("Accuracy")
                ax.set_xlabel("Illusion Quality (L=Low, M=Medium, H=High)")
                ax.set_ylim(0, 1.15)
                plt.savefig(os.path.join(charts_dir, f"{safe}_cls_accuracy_by_quality.png"))
                plt.close()

    # ── Generation charts ─────────────────────────────────────────────────
    if df_gen is not None and len(df_gen) > 0:
        # Similarity distribution box plot
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=df_gen, x="image_type", y="similarity_score", ax=ax,
                    palette=[COLORS.get(model_label, "#999")])
        ax.set_title(f"Semantic Similarity Distribution — {model_label}", fontweight="bold")
        ax.set_ylabel("Cosine Similarity (VLM guess vs label)")
        ax.set_ylim(-0.1, 1.15)
        plt.savefig(os.path.join(charts_dir, f"{safe}_gen_similarity_distribution.png"))
        plt.close()

        # Average similarity by type
        fig, ax = plt.subplots(figsize=(7, 5))
        avg_sim = df_gen.groupby("image_type")["similarity_score"].mean()
        bars = ax.bar(avg_sim.index, avg_sim.values, color=COLORS.get(model_label, "#999"), edgecolor="white", width=0.5)
        _add_bar_labels_float(ax)
        ax.set_title(f"Average Similarity by Image Type — {model_label}", fontweight="bold")
        ax.set_ylabel("Average Cosine Similarity")
        ax.set_ylim(0, 1.15)
        plt.savefig(os.path.join(charts_dir, f"{safe}_gen_similarity_by_type.png"))
        plt.close()


def generate_cross_model_charts(all_cls, all_gen, charts_dir):
    """Generate comparison charts across all models."""

    # ── Cross-model classification accuracy ───────────────────────────────
    if all_cls:
        df_all_cls = pd.concat(all_cls, ignore_index=True)

        # Grouped bar: accuracy by model and image type
        fig, ax = plt.subplots(figsize=(15, 6))
        summary = df_all_cls.groupby(["model", "image_type"])["correct"].mean().reset_index()
        summary.columns = ["Model", "Image Type", "Accuracy"]
        pivot = summary.pivot(index="Model", columns="Image Type", values="Accuracy")

        # Ensure consistent order
        model_order = [m for m in ["CLIP", "SigLIP", "ALIGN", "BLIP-2", "LLaVA-1.5", "LLaVA-1.6",
                                    "GPT-4o-mini", "GPT-5.5", "SmolVLM", "Qwen2-VL", "Moondream2"] if m in pivot.index]
        pivot = pivot.reindex(model_order)

        x = np.arange(len(pivot.index))
        width = 0.35
        cols = [c for c in ["grey", "color"] if c in pivot.columns]
        bar_colors = {"grey": "#6B7B8D", "color": "#E8734A"}
        for j, col in enumerate(cols):
            bars = ax.bar(x + j * width, pivot[col].values, width, label=col.capitalize(),
                          color=bar_colors.get(col, "#999"), edgecolor="white")
            for bar in bars:
                h = bar.get_height()
                if pd.notna(h):
                    ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2.0, h),
                                ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.set_title("Classification Accuracy: Grey vs Color — All Models", fontweight="bold", fontsize=14)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(pivot.index)
        ax.set_ylim(0, 1.2)
        ax.legend(title="Image Type")
        plt.savefig(os.path.join(charts_dir, "cross_model_classification_accuracy.png"))
        plt.close()

        # Overall accuracy per model
        fig, ax = plt.subplots(figsize=(15, 6))
        overall = df_all_cls.groupby("model")["correct"].mean().reindex(model_order)
        bar_colors_list = [COLORS.get(m, "#999") for m in model_order]
        bars = ax.bar(overall.index, overall.values, color=bar_colors_list, edgecolor="white", width=0.55)
        _add_bar_labels_decimal(ax)
        ax.set_title("Overall Classification Accuracy — All Models", fontweight="bold", fontsize=14)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.2)
        plt.savefig(os.path.join(charts_dir, "cross_model_overall_accuracy.png"))
        plt.close()

        # Grey vs Color accuracy delta per model
        fig, ax = plt.subplots(figsize=(15, 6))
        grey_acc = df_all_cls[df_all_cls["image_type"] == "grey"].groupby("model")["correct"].mean()
        color_acc = df_all_cls[df_all_cls["image_type"] == "color"].groupby("model")["correct"].mean()
        delta = (grey_acc - color_acc).reindex(model_order).fillna(0)
        bar_colors_list = [COLORS.get(m, "#999") for m in model_order]
        bars = ax.bar(delta.index, delta.values, color=bar_colors_list, edgecolor="white", width=0.55)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:+.3f}", (bar.get_x() + bar.get_width() / 2.0, h),
                        ha="center", va="bottom" if h >= 0 else "top", fontsize=10, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Grey − Color Accuracy Delta — All Models\n(Positive = better at grey)", fontweight="bold", fontsize=13)
        ax.set_ylabel("Accuracy Difference")
        plt.savefig(os.path.join(charts_dir, "cross_model_grey_vs_color_delta.png"))
        plt.close()

        # Accuracy by quality tier across models
        if df_all_cls["quality"].str.strip().ne("").any():
            fig, ax = plt.subplots(figsize=(15, 6))
            quality_order = ["L", "M", "H"]
            q_summary = df_all_cls.groupby(["model", "quality"])["correct"].mean().reset_index()
            q_summary.columns = ["Model", "Quality", "Accuracy"]
            q_pivot = q_summary.pivot(index="Model", columns="Quality", values="Accuracy")
            q_pivot = q_pivot.reindex(model_order)

            x = np.arange(len(q_pivot.index))
            width = 0.25
            q_colors = {"L": "#F1C40F", "M": "#E67E22", "H": "#E74C3C"}
            for j, q in enumerate(quality_order):
                if q in q_pivot.columns:
                    vals = q_pivot[q].values
                    bars = ax.bar(x + j * width, vals, width, label=f"Quality {q}",
                                  color=q_colors.get(q, "#999"), edgecolor="white")
                    for bar in bars:
                        h = bar.get_height()
                        if pd.notna(h):
                            ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2.0, h),
                                        ha="center", va="bottom", fontsize=8, fontweight="bold")

            ax.set_xlabel("Model")
            ax.set_ylabel("Accuracy")
            ax.set_title("Classification Accuracy by Illusion Quality — All Models", fontweight="bold", fontsize=13)
            ax.set_xticks(x + width)
            ax.set_xticklabels(q_pivot.index)
            ax.set_ylim(0, 1.25)
            ax.legend(title="Quality Tier")
            plt.savefig(os.path.join(charts_dir, "cross_model_accuracy_by_quality.png"))
            plt.close()

    # ── Cross-model generation similarity ─────────────────────────────────
    if all_gen:
        df_all_gen = pd.concat(all_gen, ignore_index=True)
        gen_model_order = [m for m in ["BLIP-2", "LLaVA-1.5", "LLaVA-1.6",
                                        "GPT-4o-mini", "GPT-5.5", "SmolVLM", "Qwen2-VL", "Moondream2"] if m in df_all_gen["model"].unique()]

        # Grouped bar: avg similarity by model and image type
        fig, ax = plt.subplots(figsize=(12, 6))
        gen_summary = df_all_gen.groupby(["model", "image_type"])["similarity_score"].mean().reset_index()
        gen_summary.columns = ["Model", "Image Type", "Similarity"]
        gen_pivot = gen_summary.pivot(index="Model", columns="Image Type", values="Similarity")
        gen_pivot = gen_pivot.reindex(gen_model_order)

        x = np.arange(len(gen_pivot.index))
        width = 0.35
        cols = [c for c in ["grey", "color"] if c in gen_pivot.columns]
        bar_colors = {"grey": "#6B7B8D", "color": "#E8734A"}
        for j, col in enumerate(cols):
            bars = ax.bar(x + j * width, gen_pivot[col].values, width, label=col.capitalize(),
                          color=bar_colors.get(col, "#999"), edgecolor="white")
            for bar in bars:
                h = bar.get_height()
                if pd.notna(h):
                    ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2.0, h),
                                ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xlabel("Model")
        ax.set_ylabel("Average Cosine Similarity")
        ax.set_title("Generation Similarity: Grey vs Color — Generative Models", fontweight="bold", fontsize=13)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(gen_pivot.index)
        ax.set_ylim(0, 1.15)
        ax.legend(title="Image Type")
        plt.savefig(os.path.join(charts_dir, "cross_model_generation_similarity.png"))
        plt.close()

        # Box plot comparison
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=df_all_gen, x="model", y="similarity_score", hue="image_type",
                    order=gen_model_order, palette={"grey": "#6B7B8D", "color": "#E8734A"}, ax=ax)
        ax.set_title("Generation Similarity Distribution — Generative Models", fontweight="bold", fontsize=13)
        ax.set_ylabel("Cosine Similarity")
        ax.set_xlabel("Model")
        ax.set_ylim(-0.1, 1.15)
        ax.legend(title="Image Type")
        plt.savefig(os.path.join(charts_dir, "cross_model_generation_boxplot.png"))
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CONSISTENT PERFORMER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_pair_number(image_name):
    """Extract the image pair number from a filename like '0002g.png' -> 2."""
    base = os.path.splitext(image_name)[0]  # '0002g'
    num_str = base[:-1]  # '0002'
    return int(num_str)


def generate_consistent_performers_report(all_cls_dfs, all_gen_dfs, raw_dir, items):
    """
    Identify image pairs that consistently performed well or poorly across
    ALL evaluated models. Saves CSV reports to raw_dir.

    For classification: a pair is 'consistently good' if every model got both
    grey and color correct; 'consistently bad' if every model got both wrong.

    For generation: uses similarity threshold (>=0.6 = good, <0.35 = bad) and
    checks consistency across all generative models.
    """
    # Build a lookup for item metadata
    item_lookup = {i["number"]: i for i in items}
    SIM_GOOD_THRESH = 0.6
    SIM_BAD_THRESH = 0.35

    # ── CLASSIFICATION ────────────────────────────────────────────────────
    if all_cls_dfs and len(all_cls_dfs) > 0:
        df_all_cls = pd.concat(all_cls_dfs, ignore_index=True)
        df_all_cls["pair_number"] = df_all_cls["image_name"].apply(_extract_pair_number)
        models = sorted(df_all_cls["model"].unique())
        n_models = len(models)

        # For each (pair_number, image_type), count how many models got it correct
        correctness = df_all_cls.groupby(["pair_number", "image_type"])["correct"].sum().reset_index()
        correctness.columns = ["pair_number", "image_type", "n_correct"]

        # Pivot so we have grey_correct and color_correct columns
        pivot = correctness.pivot(index="pair_number", columns="image_type", values="n_correct").fillna(0)
        for col in ["grey", "color"]:
            if col not in pivot.columns:
                pivot[col] = 0

        pivot["total_correct"] = pivot["grey"] + pivot["color"]
        pivot["max_possible"] = n_models * 2  # grey + color for each model
        pivot["grey_pct"] = pivot["grey"] / n_models
        pivot["color_pct"] = pivot["color"] / n_models

        # Classify each pair
        def classify_pair(row):
            if row["grey"] == n_models and row["color"] == n_models:
                return "ALL_CORRECT"
            elif row["grey"] == 0 and row["color"] == 0:
                return "ALL_WRONG"
            elif row["grey"] == n_models and row["color"] == 0:
                return "GREY_ONLY_CORRECT"
            elif row["grey"] == 0 and row["color"] == n_models:
                return "COLOR_ONLY_CORRECT"
            elif row["total_correct"] >= n_models * 2 * 0.75:
                return "MOSTLY_CORRECT"
            elif row["total_correct"] <= n_models * 2 * 0.25:
                return "MOSTLY_WRONG"
            else:
                return "MIXED"

        pivot["status"] = pivot.apply(classify_pair, axis=1)

        # Add metadata
        rows = []
        for pair_num, row in pivot.iterrows():
            meta = item_lookup.get(pair_num, {})
            rows.append({
                "pair_number": pair_num,
                "grey_object": meta.get("grey_object", ""),
                "color_object": meta.get("color_object", ""),
                "quality": meta.get("quality", ""),
                "n_models": n_models,
                "grey_correct_count": int(row["grey"]),
                "color_correct_count": int(row["color"]),
                "grey_correct_pct": f"{row['grey_pct']*100:.0f}%",
                "color_correct_pct": f"{row['color_pct']*100:.0f}%",
                "status": row["status"],
            })

        df_report = pd.DataFrame(rows)
        df_report = df_report.sort_values(["status", "pair_number"])
        cls_path = os.path.join(raw_dir, "consistent_performers_classification.csv")
        df_report.to_csv(cls_path, index=False)

        # Print summary
        counts = df_report["status"].value_counts()
        print(f"  Classification consistent performers ({n_models} models):")
        for status in ["ALL_CORRECT", "MOSTLY_CORRECT", "MIXED", "MOSTLY_WRONG",
                       "GREY_ONLY_CORRECT", "COLOR_ONLY_CORRECT", "ALL_WRONG"]:
            if status in counts.index:
                print(f"    {status}: {counts[status]} pairs")

        # Print the worst offenders
        all_wrong = df_report[df_report["status"] == "ALL_WRONG"]
        if len(all_wrong) > 0:
            print(f"  \n  Pairs ALL models got WRONG (both grey & color):")
            for _, r in all_wrong.iterrows():
                print(f"    #{r['pair_number']}: grey={r['grey_object']}, color={r['color_object']} (quality={r['quality']})")

        all_right = df_report[df_report["status"] == "ALL_CORRECT"]
        print(f"  \n  Pairs ALL models got CORRECT: {len(all_right)} pairs")
        if len(all_right) <= 20:
            for _, r in all_right.iterrows():
                print(f"    #{r['pair_number']}: grey={r['grey_object']}, color={r['color_object']} (quality={r['quality']})")

    # ── GENERATION ────────────────────────────────────────────────────────
    if all_gen_dfs and len(all_gen_dfs) > 0:
        df_all_gen = pd.concat(all_gen_dfs, ignore_index=True)
        df_all_gen["pair_number"] = df_all_gen["image_name"].apply(_extract_pair_number)
        gen_models = sorted(df_all_gen["model"].unique())
        n_gen = len(gen_models)

        # For each (pair, image_type, model) get the similarity score
        # Then for each (pair, image_type) compute avg similarity and how many models scored well/poorly
        pair_stats = df_all_gen.groupby(["pair_number", "image_type"]).agg(
            avg_sim=("similarity_score", "mean"),
            min_sim=("similarity_score", "min"),
            max_sim=("similarity_score", "max"),
            n_good=("similarity_score", lambda x: (x >= SIM_GOOD_THRESH).sum()),
            n_bad=("similarity_score", lambda x: (x < SIM_BAD_THRESH).sum()),
        ).reset_index()

        # Pivot to get grey and color side by side
        grey_stats = pair_stats[pair_stats["image_type"] == "grey"].set_index("pair_number")
        color_stats = pair_stats[pair_stats["image_type"] == "color"].set_index("pair_number")

        all_pairs = sorted(set(grey_stats.index) | set(color_stats.index))

        rows = []
        for pair_num in all_pairs:
            meta = item_lookup.get(pair_num, {})
            g = grey_stats.loc[pair_num] if pair_num in grey_stats.index else None
            c = color_stats.loc[pair_num] if pair_num in color_stats.index else None

            grey_avg = float(g["avg_sim"]) if g is not None else None
            color_avg = float(c["avg_sim"]) if c is not None else None
            grey_all_good = bool(g is not None and g["n_good"] == n_gen)
            color_all_good = bool(c is not None and c["n_good"] == n_gen)
            grey_all_bad = bool(g is not None and g["n_bad"] == n_gen)
            color_all_bad = bool(c is not None and c["n_bad"] == n_gen)

            if grey_all_good and color_all_good:
                status = "ALL_GOOD"
            elif grey_all_bad and color_all_bad:
                status = "ALL_BAD"
            elif grey_all_good and color_all_bad:
                status = "GREY_GOOD_COLOR_BAD"
            elif grey_all_bad and color_all_good:
                status = "GREY_BAD_COLOR_GOOD"
            elif (grey_avg or 0) >= SIM_GOOD_THRESH and (color_avg or 0) >= SIM_GOOD_THRESH:
                status = "MOSTLY_GOOD"
            elif (grey_avg or 1) < SIM_BAD_THRESH and (color_avg or 1) < SIM_BAD_THRESH:
                status = "MOSTLY_BAD"
            else:
                status = "MIXED"

            rows.append({
                "pair_number": pair_num,
                "grey_object": meta.get("grey_object", ""),
                "color_object": meta.get("color_object", ""),
                "quality": meta.get("quality", ""),
                "n_gen_models": n_gen,
                "grey_avg_sim": f"{grey_avg:.3f}" if grey_avg is not None else "",
                "color_avg_sim": f"{color_avg:.3f}" if color_avg is not None else "",
                "grey_min_sim": f"{float(g['min_sim']):.3f}" if g is not None else "",
                "color_min_sim": f"{float(c['min_sim']):.3f}" if c is not None else "",
                "status": status,
            })

        df_gen_report = pd.DataFrame(rows)
        df_gen_report = df_gen_report.sort_values(["status", "pair_number"])
        gen_path = os.path.join(raw_dir, "consistent_performers_generation.csv")
        df_gen_report.to_csv(gen_path, index=False)

        # Print summary
        gen_counts = df_gen_report["status"].value_counts()
        print(f"\n  Generation consistent performers ({n_gen} models, good>={SIM_GOOD_THRESH}, bad<{SIM_BAD_THRESH}):")
        for status in ["ALL_GOOD", "MOSTLY_GOOD", "MIXED", "MOSTLY_BAD",
                       "GREY_GOOD_COLOR_BAD", "GREY_BAD_COLOR_GOOD", "ALL_BAD"]:
            if status in gen_counts.index:
                print(f"    {status}: {gen_counts[status]} pairs")

        all_bad_gen = df_gen_report[df_gen_report["status"] == "ALL_BAD"]
        if len(all_bad_gen) > 0:
            print(f"  \n  Pairs ALL gen models scored poorly on:")
            for _, r in all_bad_gen.iterrows():
                print(f"    #{r['pair_number']}: grey={r['grey_object']}(sim={r['grey_avg_sim']}), "
                      f"color={r['color_object']}(sim={r['color_avg_sim']}) quality={r['quality']}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _run_model_safe(name, fn, all_cls, all_gen, raw_dir, charts_dir, **kwargs):
    """Run a model evaluator wrapped in try/except. If it fails, skip it."""
    print(f"\n  ▶ Starting {name}...")
    try:
        result = fn(**kwargs)
        # Handle different return signatures
        if isinstance(result, tuple) and len(result) == 2:
            df_cls, df_gen = result
        else:
            df_cls, df_gen = result, pd.DataFrame()

        safe = name.replace(" ", "_").replace("-", "_").replace(".", "")
        if df_cls is not None and len(df_cls) > 0:
            df_cls.to_csv(os.path.join(raw_dir, f"{safe}_classification.csv"), index=False)
            generate_per_model_charts(df_cls, df_gen if len(df_gen) > 0 else None, name, charts_dir)
            all_cls.append(df_cls)
            acc = df_cls['correct'].mean() * 100
            msg = f"  ✓ {name} done — cls acc: {acc:.1f}%"
            if df_gen is not None and len(df_gen) > 0:
                df_gen.to_csv(os.path.join(raw_dir, f"{safe}_generation.csv"), index=False)
                all_gen.append(df_gen)
                msg += f", gen sim: {df_gen['similarity_score'].mean():.3f}"
            print(msg)
        else:
            print(f"  ⚠ {name} returned no classification results.")
    except Exception as e:
        print(f"\n  ✗ {name} FAILED: {type(e).__name__}: {e}")
        print(f"    Skipping {name} and continuing...\n")
        # Force cleanup on failure
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    _style_setup()
    device = get_device()
    print(f"Device: {device}")

    # Create output directories
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("master_results", timestamp)
    raw_dir = os.path.join(run_dir, "raw_data")
    charts_dir = os.path.join(run_dir, "charts")
    reports_dir = os.path.join(run_dir, "reports")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    print(f"Output directory: {run_dir}")

    items = load_data()

    all_cls_dfs = []   # classification DataFrames
    all_gen_dfs = []   # generation DataFrames

    # ── 1. CLIP ───────────────────────────────────────────────────────────
    if RUN_CLIP:
        _run_model_safe("CLIP", lambda: eval_clip_classification(items, device),
                        all_cls_dfs, all_gen_dfs, raw_dir, charts_dir)

    # ── 2. SigLIP ─────────────────────────────────────────────────────────
    if RUN_SIGLIP:
        _run_model_safe("SigLIP", lambda: eval_siglip_classification(items, device),
                        all_cls_dfs, all_gen_dfs, raw_dir, charts_dir)

    # ── 3. ALIGN ──────────────────────────────────────────────────────────
    if RUN_ALIGN:
        _run_model_safe("ALIGN", lambda: eval_align_classification(items, device),
                        all_cls_dfs, all_gen_dfs, raw_dir, charts_dir)

    # ── 4. BLIP-2 ─────────────────────────────────────────────────────────
    if RUN_BLIP2:
        _run_model_safe("BLIP-2", lambda: run_blip2(items, device),
                        all_cls_dfs, all_gen_dfs, raw_dir, charts_dir)

    # ── 4. LLaVA 1.5 ─────────────────────────────────────────────────────
    if RUN_LLAVA_15:
        _run_model_safe("LLaVA-1.5", lambda: run_llava15(items, device),
                        all_cls_dfs, all_gen_dfs, raw_dir, charts_dir)

    # ── 5. LLaVA 1.6 ─────────────────────────────────────────────────────
    if RUN_LLAVA_16:
        _run_model_safe("LLaVA-1.6", lambda: run_llava16(items, device),
                        all_cls_dfs, all_gen_dfs, raw_dir, charts_dir)

    # ── 6. GPT-4o-mini (OpenAI API) ──────────────────────────────────────
    if RUN_GPT4O_MINI:
        from new_models import run_gpt4o_mini
        _run_model_safe("GPT-4o-mini",
                        lambda: run_gpt4o_mini(items, IMAGE_DIR, SIM_MODEL),
                        all_cls_dfs, all_gen_dfs, raw_dir, charts_dir)

    # ── 7. GPT-5.5 (OpenAI API) ──────────────────────────────────────────
    if RUN_GPT55:
        from new_models import run_gpt55
        _run_model_safe("GPT-5.5",
                        lambda: run_gpt55(items, IMAGE_DIR, SIM_MODEL),
                        all_cls_dfs, all_gen_dfs, raw_dir, charts_dir)

    # ── 8. SmolVLM ───────────────────────────────────────────────────────
    if RUN_SMOLVLM:
        from new_models import run_smolvlm
        _run_model_safe("SmolVLM",
                        lambda: run_smolvlm(items, device, IMAGE_DIR, SIM_MODEL),
                        all_cls_dfs, all_gen_dfs, raw_dir, charts_dir)

    # ── 8. Qwen2-VL ──────────────────────────────────────────────────────
    if RUN_QWEN2VL:
        from new_models import run_qwen2vl
        _run_model_safe("Qwen2-VL",
                        lambda: run_qwen2vl(items, device, IMAGE_DIR, SIM_MODEL),
                        all_cls_dfs, all_gen_dfs, raw_dir, charts_dir)

    # ── 9. Moondream2 ────────────────────────────────────────────────────
    if RUN_MOONDREAM2:
        from new_models import run_moondream2
        _run_model_safe("Moondream2",
                        lambda: run_moondream2(items, device, IMAGE_DIR, SIM_MODEL),
                        all_cls_dfs, all_gen_dfs, raw_dir, charts_dir)



    # ── Cross-model comparison charts ─────────────────────────────────────
    print("\nGenerating cross-model comparison charts...")
    try:
        generate_cross_model_charts(all_cls_dfs, all_gen_dfs, charts_dir)
    except Exception as e:
        print(f"  ⚠ Cross-model charts failed: {e}")

    # ── Consistent performer analysis ─────────────────────────────────────
    print("\nAnalyzing consistent performers across models...")
    try:
        generate_consistent_performers_report(all_cls_dfs, all_gen_dfs, raw_dir, items)
    except Exception as e:
        print(f"  ⚠ Consistent performers report failed: {e}")

    # ── Detailed reports ──────────────────────────────────────────────────
    print("\nGenerating detailed reports...")
    try:
        from reports_generator import generate_all_reports
        generate_all_reports(all_cls_dfs, all_gen_dfs, items, reports_dir)
    except Exception as e:
        print(f"  ⚠ Reports generation failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ALL DONE — results in: {run_dir}")
    print(f"{'='*60}")

    if all_cls_dfs:
        df_all = pd.concat(all_cls_dfs, ignore_index=True)
        summary = df_all.groupby(["model", "image_type"])["correct"].mean().unstack()
        summary["overall"] = df_all.groupby("model")["correct"].mean()
        print("\nClassification Accuracy Summary:")
        print(summary.to_string(float_format=lambda x: f"{x*100:.1f}%"))

    if all_gen_dfs:
        df_all_gen = pd.concat(all_gen_dfs, ignore_index=True)
        gen_summary = df_all_gen.groupby(["model", "image_type"])["similarity_score"].mean().unstack()
        gen_summary["overall"] = df_all_gen.groupby("model")["similarity_score"].mean()
        print("\nGeneration Similarity Summary:")
        print(gen_summary.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
