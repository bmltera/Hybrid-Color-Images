r"""
test_new_models.py
Smoke-test each of the 5 new VLM models on a single image pair.
Run with:  .\venv\Scripts\python.exe test_new_models.py
"""

import os, sys, json, time, gc, traceback
import torch
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_DIR = "data"
DATA_JSON = "1.json"
SIM_MODEL = "all-MiniLM-L6-v2"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_test_item():
    """Load just ONE item from the dataset for testing."""
    with open(DATA_JSON) as f:
        data = json.load(f)["data"]
    for item in data:
        if item.get("grey_object") and item.get("color_object"):
            num = f"{item['number']:04d}"
            gp = os.path.join(IMAGE_DIR, f"{num}g.png")
            cp = os.path.join(IMAGE_DIR, f"{num}c.png")
            if os.path.exists(gp) and os.path.exists(cp):
                return item
    raise RuntimeError("No valid test item found")

def gpu_mem():
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{used:.1f}/{total:.1f} GB"
    return "N/A"

# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

results = {}

def test_model(name, fn):
    """Run a model test, catch errors, record PASS/FAIL."""
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"  GPU before: {gpu_mem()}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        results[name] = "PASS"
        print(f"\n  >> {name}: PASS ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        results[name] = f"FAIL: {type(e).__name__}: {e}"
        print(f"\n  >> {name}: FAIL ({elapsed:.1f}s)")
        traceback.print_exc()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  GPU after cleanup: {gpu_mem()}")


# ── 1. GPT-4o-mini (OpenAI API) ───────────────────────────────────────────────────
def test_gpt4o_mini():
    import base64
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    assert api_key and api_key != "your_key_here", "OPENAI_API_KEY not set in .env"

    client = OpenAI(api_key=api_key)
    item = load_test_item()
    num = f"{item['number']:04d}"
    img_path = os.path.join(IMAGE_DIR, f"{num}g.png")
    go, co = item["grey_object"], item["color_object"]

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    # Classification
    print("  Loading GPT-4o-mini via OpenAI API...")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": f"Is this a {go} or a {co}? Reply with ONLY the object name."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
            ]}],
            max_completion_tokens=20,
        )
        answer = resp.choices[0].message.content.strip()
        print(f"  CLS answer: '{answer}' (expected: {go})")
        assert len(answer) > 0, "Empty response from GPT-4o-mini"

        # Generation
        resp2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "What is the main object in this image? Reply with just one specific noun."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
            ]}],
            max_completion_tokens=20,
        )
        guess = resp2.choices[0].message.content.strip()
        print(f"  GEN guess:  '{guess}'")
        assert len(guess) > 0, "Empty generation from GPT-4o-mini"
    except Exception as e:
        print(f"  API Error: {e}")
        raise

# ── 2. GPT-5.5 (OpenAI API) ───────────────────────────────────────────────────
def test_gpt55():
    import base64
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    assert api_key and api_key != "your_key_here", "OPENAI_API_KEY not set in .env"

    client = OpenAI(api_key=api_key)
    item = load_test_item()
    num = f"{item['number']:04d}"
    img_path = os.path.join(IMAGE_DIR, f"{num}g.png")
    go, co = item["grey_object"], item["color_object"]

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    # Classification
    print("  Loading GPT-5.5 via OpenAI API...")
    try:
        resp = client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": f"Is this a {go} or a {co}? Reply with ONLY the object name."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
            ]}],
            max_completion_tokens=20,
        )
        answer = resp.choices[0].message.content.strip()
        print(f"  CLS answer: '{answer}' (expected: {go})")
        assert len(answer) > 0, "Empty response from GPT-5.5"

        # Generation
        resp2 = client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "What is the main object in this image? Reply with just one specific noun."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
            ]}],
            max_completion_tokens=20,
        )
        guess = resp2.choices[0].message.content.strip()
        print(f"  GEN guess:  '{guess}'")
        assert len(guess) > 0, "Empty generation from GPT-5.5"
    except Exception as e:
        print(f"  API Error: {e}")
        raise
# ── 2. SmolVLM ────────────────────────────────────────────────────────────────
def test_smolvlm():
    from transformers import AutoProcessor, AutoModelForImageTextToText

    device = get_device()
    model_name = "HuggingFaceTB/SmolVLM-Instruct"
    print(f"  Loading {model_name}...")

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    ).eval()
    print(f"  Model loaded. GPU: {gpu_mem()}")

    item = load_test_item()
    num = f"{item['number']:04d}"
    img_path = os.path.join(IMAGE_DIR, f"{num}g.png")
    go, co = item["grey_object"], item["color_object"]
    image = Image.open(img_path).convert("RGB")

    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": f"Is this a {go} or a {co}? Reply with just the object name."},
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    trimmed = out[0][inputs["input_ids"].shape[1]:]
    answer = processor.decode(trimmed, skip_special_tokens=True).strip()
    print(f"  CLS answer: '{answer}'")
    assert len(answer) > 0, "Empty answer from SmolVLM"

    # Cleanup
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

# ── 3. Qwen2-VL ──────────────────────────────────────────────────────────────
def test_qwen2vl():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    device = get_device()
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    print(f"  Loading {model_name}...")

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    ).eval()
    print(f"  Model loaded. GPU: {gpu_mem()}")

    item = load_test_item()
    num = f"{item['number']:04d}"
    img_path = os.path.join(IMAGE_DIR, f"{num}g.png")
    go, co = item["grey_object"], item["color_object"]
    image = Image.open(img_path).convert("RGB")

    # Build prompt
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": f"Is this a {go} or a {co}? Reply with just the object name."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    trimmed = out[0][inputs["input_ids"].shape[1]:]
    answer = processor.decode(trimmed, skip_special_tokens=True).strip()
    print(f"  CLS answer: '{answer}'")
    assert len(answer) > 0, "Empty answer from Qwen2-VL"

    # Cleanup
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


# -- 4. Moondream2 ----------------------------------------------------------------
def test_moondream2():
    from transformers import AutoModelForCausalLM

    device = get_device()
    model_name = "vikhyatk/moondream2"
    print(f"  Loading {model_name}...")

    # Monkey-patch for transformers 5.x
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
    print(f"  Model loaded. GPU: {gpu_mem()}")

    item = load_test_item()
    num = f"{item['number']:04d}"
    img_path = os.path.join(IMAGE_DIR, f"{num}g.png")
    go, co = item["grey_object"], item["color_object"]
    image = Image.open(img_path).convert("RGB")

    # Classification via query()
    answer = model.query(image, f"Is this a {go} or a {co}? Reply with just the object name.")["answer"]
    print(f"  CLS answer: '{answer}'")
    assert len(answer.strip()) > 0, "Empty answer from Moondream2"

    # Generation via caption()
    caption = model.caption(image, length="short")["caption"]
    print(f"  Caption:    '{caption}'")
    assert len(caption.strip()) > 0, "Empty caption from Moondream2"

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()



# ── 5. ALIGN ─────────────────────────────────────────────────────────────
def test_align():
    from transformers import AlignProcessor, AlignModel
    device = get_device()
    model_name = "kakaobrain/align-base"
    print(f"  Loading {model_name}...")
    processor = AlignProcessor.from_pretrained(model_name)
    model = AlignModel.from_pretrained(model_name).to(device)
    print(f"  Model loaded. GPU: {gpu_mem()}")

    item = load_test_item()
    num = f"{item['number']:04d}"
    img_path = os.path.join(IMAGE_DIR, f"{num}g.png")
    go, co = item["grey_object"], item["color_object"]
    image = Image.open(img_path).convert("RGB")

    labels = [f"a photo of a {go}", f"a photo of a {co}"]
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
    pred_is_grey = probs[0] > probs[1]
    pred_obj = go if pred_is_grey else co
    print(f"  CLS answer: '{pred_obj}' (expected: {go})")
    
    # Cleanup
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  VLM MODEL SMOKE TESTS")
    print(f"  Device: {get_device()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    test_model("ALIGN",            test_align)
    test_model("GPT-4o-mini",      test_gpt4o_mini)
    test_model("GPT-5.5",          test_gpt55)
    test_model("SmolVLM",          test_smolvlm)
    test_model("Qwen2-VL-7B",      test_qwen2vl)
    test_model("Moondream2",       test_moondream2)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, status in results.items():
        icon = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {icon} {name}: {status}")
        if status != "PASS":
            all_pass = False

    if all_pass:
        print(f"\n  All {len(results)} models passed!")
    else:
        failed = [n for n, s in results.items() if s != "PASS"]
        print(f"\n  {len(failed)}/{len(results)} models failed: {', '.join(failed)}")

    sys.exit(0 if all_pass else 1)
