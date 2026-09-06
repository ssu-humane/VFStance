# VFStance
[![arXiv](https://img.shields.io/badge/arXiv-2609.00685-b31b1b.svg?style=flat)](https://arxiv.org/abs/2609.00685)

Official repository for **"Visual Framing for News Stance Detection via Generated Images"**.

VFStanceGen is a three-stage framework that leverages visual framing theory to generate stance-aware images and predict article-level news stance:

1. **Stage 1 — Visual Framing Annotation**: an LLM annotates an article into a structured visual-framing specification (ideological, connotative, stylistic-semiotic, denotative).
2. **Stage 2 — Stance-Aware Image Generation**: a T2I model generates an image from the specification.
3. **Stage 3 — Stance Prediction**: an LVLM predicts the stance from the article text and the generated image.

## Repository Structure

- `vfstancegen/` — main pipeline (Stages 1–3); includes `vfstancegen_open.py` for running the full pipeline with open-weight models
- `baseline/` — textual, visual, multimodal, and LVLM baselines
- `eaig4sd/` — EAIG4SD baseline reproduction
- `evaluation/` — metric aggregation and significance testing
- `prompts/` — prompt templates for all stages

## Setup

```bash
pip install -r requirements.txt
```

Place the dataset under `data/` and create a `.env` file with your API keys (see `README` in each directory for details).

## Running

```bash
python vfstancegen/stage1_annotate_visual_framing.py
python vfstancegen/stage2_generate_image.py
python vfstancegen/stage3_predict_stance.py

# or with open-weight models:
python vfstancegen/vfstancegen_open.py --stage all
```

Dataset will be released upon publication.
