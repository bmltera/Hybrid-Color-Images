# Entity Recognition with Vision Language Models on Diffusion-Based Color Hybrid Illusions

This repository contains the code, data, and interactive project website for our research on evaluating vision-language models (VLMs) using color hybrid illusions generated with Factorized Diffusion.

**Authors:** Bill Li, Paul Junver Soriano, Rahul Koonantavida
**Affiliation:** San Jose State University, College of Engineering

## Overview

Vision-language models perform well on standard multimodal benchmarks, but it is unclear which visual cues they actually rely on. We study this using color hybrid images, where the color view of an image depicts one entity and the grayscale view depicts a completely different entity. This creates a controlled cue conflict that lets us measure whether a model favors chromatic (color) or luminance (grayscale) information.

We generated 2,400 candidate image pairs using Factorized Diffusion and manually audited them down to a final benchmark of 177 high-quality pairs. We then evaluated 11 VLMs across 5 architecture families, producing 3,894 total predictions.

**Key finding:** Most VLMs exhibit a grayscale bias, achieving 68.1% accuracy on grayscale targets vs. 55.4% on color targets. This 12.7 percentage point gap suggests that many models rely more on shape and structure than on color when the two conflict.

## Links

| Resource | URL |
|----------|-----|
| Project Website | https://hybrid-color-images.vercel.app/ |
| Dataset (HuggingFace) | https://huggingface.co/datasets/bmltera/color-hybrid-illusions |
| Project Artifacts (SJSU) | https://drive.google.com/drive/folders/1WOsYUHJopPhN9KRBTWWvJ2kxmF5EImUg?usp=sharing |

## Repository Structure

```
.
├── notebook/                    # Image generation scripts and notebooks
│   ├── generate.py              # Main generation script
│   ├── generation*.ipynb        # Jupyter notebooks for generation runs
│   └── prompt_pairs_*.json      # Prompt pair definitions
│
├── cv_ui/                       # Project website
    ├── frontend/             
    ├── data/                    # Raw evaluation data and generated charts
        ├── raw_data/            # CSV results from model evaluations
        ├── charts/              # Generated chart images
        └── reports/             # Analysis reports
```


### Image Generation

The generation pipeline requires a CUDA-capable GPU and uses DeepFloyd IF through the Factorized Diffusion framework.

```bash
cd notebook
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Then run generation:

```bash
python generate.py
```

Or use `generate_resume.py` to resume from a specific index if the process gets interrupted.

### Project Website

The website is built with React and Vite. To run locally:

```bash
cd cv_ui/frontend
npm install
npm run dev
```

The site is deployed at https://hybrid-color-images.vercel.app/.

### Dataset

The curated dataset of 177 image pairs is available on HuggingFace:

https://huggingface.co/datasets/bmltera/color-hybrid-illusions

Each entry includes the color image, grayscale image, entity labels for both views, and a human-rated quality tier (L/M/H).

## Citation

If you use this benchmark or dataset, please cite:

```bibtex
@inproceedings{li2026entity,
  title={Entity Recognition with Vision Language Models on Diffusion-Based Color Hybrid Illusions},
  author={Li, Bill and Soriano, Paul Junver and Koonantavida, Rahul},
  year={2026}
}
```

## Acknowledgments

This project builds on the Factorized Diffusion framework by Geng et al. (ECCV 2024) and the Visual Anagrams codebase. We also reference MMStar, VIA-Bench, and CLIPScore in our evaluation design. Full references are available in the paper.
