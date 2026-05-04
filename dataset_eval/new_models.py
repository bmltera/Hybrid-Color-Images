"""
new_models.py — Additional VLM evaluators for the master eval pipeline.

Models:
  6. GPT-4o-mini          (OpenAI API)       — gen + cls
  7. GPT-5.5              (OpenAI API)       — gen + cls
  7. Phi-3.5-Vision       (microsoft)        — gen + cls  (~8 GB)
  8. Qwen2-VL-7B          (Alibaba)          — gen + cls  (~16 GB)
  9. Moondream2            (vikhyatk)         — gen + cls  (~4 GB)
"""

import os, gc, time
import pandas as pd
import torch
from PIL import Image

# ── shared helpers (imported by master_eval) ──────────────────────────────────

def _get_image_paths(item, image_dir):
    num = f"{item['number']:04d}"
    return os.path.join(image_dir, f"{num}g.png"), os.path.join(image_dir, f"{num}c.png")


def _free_model(*models):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for m in models:
        if hasattr(m, "cpu"):
            try:
                m.cpu()
            except Exception:
                pass
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ═════════════════════════════════════════════════════════════════════════════
#  6. GPT-5.5 Pro  (OpenAI API)
# ═════════════════════════════════════════════════════════════════════════════

def _run_gpt(items, image_dir, model_id, label, sim_model_name="all-MiniLM-L6-v2"):
    import base64
    try:
        from openai import OpenAI
    except ImportError:
        print("  ERROR: 'openai' package not installed. pip install openai")
        return pd.DataFrame(), pd.DataFrame()
    from sentence_transformers import SentenceTransformer, util
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        print(f"  ERROR: OPENAI_API_KEY not set. Skipping {label}.")
        return pd.DataFrame(), pd.DataFrame()

    client = OpenAI(api_key=api_key)
    sim_model = SentenceTransformer(sim_model_name, device="cpu")
    print(f"\n{'='*60}\n  {label}: OpenAI API ({model_id})\n{'='*60}")

    cls_results, gen_results = [], []
    total = len(items)

    for i, item in enumerate(items, 1):
        gp, cp = _get_image_paths(item, image_dir)
        go, co = item["grey_object"], item["color_object"]

        for img_path, true_obj, img_type in [(gp, go, "grey"), (cp, co, "color")]:
            if not os.path.exists(img_path):
                continue
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            img_url = f"data:image/png;base64,{b64}"

            # ── classification ───────────────────────────────────────
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": f"Is this a {go} or a {co}? Reply with ONLY the object name, nothing else."},
                        {"type": "image_url", "image_url": {"url": img_url, "detail": "low"}},
                    ]}],
                    max_completion_tokens=20,
                )
                answer = resp.choices[0].message.content.strip().lower()
                ea = sim_model.encode(answer, convert_to_tensor=True)
                eg = sim_model.encode(go, convert_to_tensor=True)
                ec = sim_model.encode(co, convert_to_tensor=True)
                sg = util.pytorch_cos_sim(ea, eg).item()
                sc = util.pytorch_cos_sim(ea, ec).item()
                pred = go if sg > sc else co
                cls_results.append({
                    "model": label, "image_name": os.path.basename(img_path),
                    "image_type": img_type, "true_object": true_obj,
                    "pred_object": pred, "vlm_answer": answer,
                    "sim_to_grey": sg, "sim_to_color": sc,
                    "correct": pred == true_obj, "quality": item.get("quality", ""),
                })
            except Exception as e:
                print(f"    cls error on {img_path}: {e}")

            # ── generation ───────────────────────────────────────────
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": "What is the main object in this image? Reply with just one specific noun."},
                        {"type": "image_url", "image_url": {"url": img_url, "detail": "low"}},
                    ]}],
                    max_completion_tokens=20,
                )
                guess = resp.choices[0].message.content.strip().lower()
                eg2 = sim_model.encode(guess, convert_to_tensor=True)
                et = sim_model.encode(true_obj, convert_to_tensor=True)
                cos = util.pytorch_cos_sim(eg2, et).item()
                gen_results.append({
                    "model": label, "image_name": os.path.basename(img_path),
                    "image_type": img_type, "true_object": true_obj,
                    "vlm_guess": guess, "similarity_score": cos,
                    "quality": item.get("quality", ""),
                })
            except Exception as e:
                print(f"    gen error on {img_path}: {e}")

        if i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] {i/total*100:.1f}%")

    del sim_model
    return pd.DataFrame(cls_results), pd.DataFrame(gen_results)

def run_gpt4o_mini(items, image_dir, sim_model_name="all-MiniLM-L6-v2"):
    """Run GPT-4o-mini via OpenAI API."""
    return _run_gpt(items, image_dir, "gpt-4o-mini", "GPT-4o-mini", sim_model_name)

def run_gpt55(items, image_dir, sim_model_name="all-MiniLM-L6-v2"):
    """Run GPT-5.5 via OpenAI API."""
    return _run_gpt(items, image_dir, "gpt-5.5", "GPT-5.5", sim_model_name)


# ═════════════════════════════════════════════════════════════════════════════
#  7. SmolVLM-Instruct  (replaces Florence-2 & Phi-3.5 — broken in transformers 5.x)
# ═════════════════════════════════════════════════════════════════════════════

def run_smolvlm(items, device, image_dir, sim_model_name="all-MiniLM-L6-v2"):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from sentence_transformers import SentenceTransformer, util

    model_name = "HuggingFaceTB/SmolVLM-Instruct"
    label = "SmolVLM"
    print(f"\n{'='*60}\n  {label}: {model_name}\n{'='*60}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    ).eval()
    sim_model = SentenceTransformer(sim_model_name, device="cpu")

    cls_results, gen_results = [], []
    total = len(items)

    def _ask(image, question):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question},
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        return processor.decode(trimmed, skip_special_tokens=True).strip().lower()

    for i, item in enumerate(items, 1):
        gp, cp = _get_image_paths(item, image_dir)
        go, co = item["grey_object"], item["color_object"]

        for img_path, true_obj, img_type in [(gp, go, "grey"), (cp, co, "color")]:
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")

            # classification
            answer = _ask(image, f"Is this a {go} or a {co}? Reply with just the object name.")
            ea = sim_model.encode(answer, convert_to_tensor=True)
            eg = sim_model.encode(go, convert_to_tensor=True)
            ec = sim_model.encode(co, convert_to_tensor=True)
            sg = util.pytorch_cos_sim(ea, eg).item()
            sc = util.pytorch_cos_sim(ea, ec).item()
            pred = go if sg > sc else co
            cls_results.append({
                "model": label, "image_name": os.path.basename(img_path),
                "image_type": img_type, "true_object": true_obj,
                "pred_object": pred, "vlm_answer": answer,
                "sim_to_grey": sg, "sim_to_color": sc,
                "correct": pred == true_obj, "quality": item.get("quality", ""),
            })

            # generation
            guess = _ask(image, "What is the main object in this image? Reply with just one specific noun.")
            et = sim_model.encode(true_obj, convert_to_tensor=True)
            eg2 = sim_model.encode(guess, convert_to_tensor=True)
            cos = util.pytorch_cos_sim(eg2, et).item()
            gen_results.append({
                "model": label, "image_name": os.path.basename(img_path),
                "image_type": img_type, "true_object": true_obj,
                "vlm_guess": guess, "similarity_score": cos,
                "quality": item.get("quality", ""),
            })

        if i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] {i/total*100:.1f}%")

    _free_model(model, processor, sim_model)
    return pd.DataFrame(cls_results), pd.DataFrame(gen_results)


# ═════════════════════════════════════════════════════════════════════════════
#  8. Qwen2-VL-7B-Instruct
# ═════════════════════════════════════════════════════════════════════════════

def run_qwen2vl(items, device, image_dir, sim_model_name="all-MiniLM-L6-v2"):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from sentence_transformers import SentenceTransformer, util

    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    label = "Qwen2-VL"
    print(f"\n{'='*60}\n  {label}: {model_name}\n{'='*60}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    ).eval()
    sim_model = SentenceTransformer(sim_model_name, device="cpu")

    cls_results, gen_results = [], []
    total = len(items)

    def _ask(image, text_prompt):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": text_prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        return processor.decode(trimmed, skip_special_tokens=True).strip().lower()

    for i, item in enumerate(items, 1):
        gp, cp = _get_image_paths(item, image_dir)
        go, co = item["grey_object"], item["color_object"]

        for img_path, true_obj, img_type in [(gp, go, "grey"), (cp, co, "color")]:
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")

            # classification
            answer = _ask(image, f"Is this a {go} or a {co}? Reply with just the object name.")
            ea = sim_model.encode(answer, convert_to_tensor=True)
            eg = sim_model.encode(go, convert_to_tensor=True)
            ec = sim_model.encode(co, convert_to_tensor=True)
            sg = util.pytorch_cos_sim(ea, eg).item()
            sc = util.pytorch_cos_sim(ea, ec).item()
            pred = go if sg > sc else co
            cls_results.append({
                "model": label, "image_name": os.path.basename(img_path),
                "image_type": img_type, "true_object": true_obj,
                "pred_object": pred, "vlm_answer": answer,
                "sim_to_grey": sg, "sim_to_color": sc,
                "correct": pred == true_obj, "quality": item.get("quality", ""),
            })

            # generation
            guess = _ask(image, "What is the main object in this image? Reply with just one specific noun.")
            et = sim_model.encode(true_obj, convert_to_tensor=True)
            eg2 = sim_model.encode(guess, convert_to_tensor=True)
            cos = util.pytorch_cos_sim(eg2, et).item()
            gen_results.append({
                "model": label, "image_name": os.path.basename(img_path),
                "image_type": img_type, "true_object": true_obj,
                "vlm_guess": guess, "similarity_score": cos,
                "quality": item.get("quality", ""),
            })

        if i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] {i/total*100:.1f}%")

    _free_model(model, processor, sim_model)
    return pd.DataFrame(cls_results), pd.DataFrame(gen_results)


# ═════════════════════════════════════════════════════════════════════════════
#  9. Moondream2  (replaces PaliGemma — gated repo)
# ═════════════════════════════════════════════════════════════════════════════

def run_moondream2(items, device, image_dir, sim_model_name="all-MiniLM-L6-v2"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer, util

    model_name = "vikhyatk/moondream2"
    label = "Moondream2"
    print(f"\n{'='*60}\n  {label}: {model_name}\n{'='*60}")

    # Monkey-patch for transformers 5.x compatibility
    _orig_getattr = torch.nn.Module.__getattr__
    def _patched_getattr(self, name):
        if name == "all_tied_weights_keys":
            return {}
        return _orig_getattr(self, name)
    torch.nn.Module.__getattr__ = _patched_getattr

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            device_map={"": device},
        ).eval()
    finally:
        torch.nn.Module.__getattr__ = _orig_getattr
    sim_model = SentenceTransformer(sim_model_name, device="cpu")

    cls_results, gen_results = [], []
    total = len(items)

    for i, item in enumerate(items, 1):
        gp, cp = _get_image_paths(item, image_dir)
        go, co = item["grey_object"], item["color_object"]

        for img_path, true_obj, img_type in [(gp, go, "grey"), (cp, co, "color")]:
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")

            # classification via query()
            cls_q = f"Is this a {go} or a {co}? Reply with just the object name."
            answer = model.query(image, cls_q)["answer"].strip().lower()

            ea = sim_model.encode(answer, convert_to_tensor=True)
            eg = sim_model.encode(go, convert_to_tensor=True)
            ec = sim_model.encode(co, convert_to_tensor=True)
            sg = util.pytorch_cos_sim(ea, eg).item()
            sc = util.pytorch_cos_sim(ea, ec).item()
            pred = go if sg > sc else co
            cls_results.append({
                "model": label, "image_name": os.path.basename(img_path),
                "image_type": img_type, "true_object": true_obj,
                "pred_object": pred, "vlm_answer": answer,
                "sim_to_grey": sg, "sim_to_color": sc,
                "correct": pred == true_obj, "quality": item.get("quality", ""),
            })

            # generation via caption()
            caption = model.caption(image, length="short")["caption"].strip().lower()
            et = sim_model.encode(true_obj, convert_to_tensor=True)
            ec2 = sim_model.encode(caption, convert_to_tensor=True)
            cos = util.pytorch_cos_sim(ec2, et).item()
            gen_results.append({
                "model": label, "image_name": os.path.basename(img_path),
                "image_type": img_type, "true_object": true_obj,
                "vlm_guess": caption, "similarity_score": cos,
                "quality": item.get("quality", ""),
            })

        if i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] {i/total*100:.1f}%")

    _free_model(model, sim_model)
    return pd.DataFrame(cls_results), pd.DataFrame(gen_results)


# ═════════════════════════════════════════════════════════════════════════════
#  10. LLaVA-NeXT (Mistral-7B)  (replaces InternVL2 — broken in transformers 5.x)
# ═════════════════════════════════════════════════════════════════════════════

def run_llava_next(items, device, image_dir, sim_model_name="all-MiniLM-L6-v2"):
    from transformers import LlavaNextForConditionalGeneration, AutoProcessor
    from sentence_transformers import SentenceTransformer, util

    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    label = "LLaVA-NeXT"
    print(f"\n{'='*60}\n  {label}: {model_name}\n{'='*60}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    ).eval()
    sim_model = SentenceTransformer(sim_model_name, device="cpu")

    cls_results, gen_results = [], []
    total = len(items)

    def _ask(image, question):
        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question},
        ]}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        return processor.decode(trimmed, skip_special_tokens=True).strip().lower()

    for i, item in enumerate(items, 1):
        gp, cp = _get_image_paths(item, image_dir)
        go, co = item["grey_object"], item["color_object"]

        for img_path, true_obj, img_type in [(gp, go, "grey"), (cp, co, "color")]:
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")

            # classification
            answer = _ask(image, f"Is this a {go} or a {co}? Reply with just the object name.")
            ea = sim_model.encode(answer, convert_to_tensor=True)
            eg = sim_model.encode(go, convert_to_tensor=True)
            ec = sim_model.encode(co, convert_to_tensor=True)
            sg = util.pytorch_cos_sim(ea, eg).item()
            sc = util.pytorch_cos_sim(ea, ec).item()
            pred = go if sg > sc else co
            cls_results.append({
                "model": label, "image_name": os.path.basename(img_path),
                "image_type": img_type, "true_object": true_obj,
                "pred_object": pred, "vlm_answer": answer,
                "sim_to_grey": sg, "sim_to_color": sc,
                "correct": pred == true_obj, "quality": item.get("quality", ""),
            })

            # generation
            guess = _ask(image, "What is the main object in this image? Reply with just one specific noun.")
            et = sim_model.encode(true_obj, convert_to_tensor=True)
            eg2 = sim_model.encode(guess, convert_to_tensor=True)
            cos = util.pytorch_cos_sim(eg2, et).item()
            gen_results.append({
                "model": label, "image_name": os.path.basename(img_path),
                "image_type": img_type, "true_object": true_obj,
                "vlm_guess": guess, "similarity_score": cos,
                "quality": item.get("quality", ""),
            })

        if i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] {i/total*100:.1f}%")

    _free_model(model, processor, sim_model)
    return pd.DataFrame(cls_results), pd.DataFrame(gen_results)



