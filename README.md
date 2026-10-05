# VFStance

Official implementation of **Visual Framing for News Stance Detection via Image Generation** ([paper](https://arxiv.org/abs/2609.00685)).

The repository includes:

- `vfstance`: VFStance and VFStance (Text)
- `baseline`: textual, visual, multimodal, and LVLM baselines
- `eaig4sd`: the image-generation and selection baseline
- `evaluation`: metrics, run aggregation, and significance testing

## Installation

Install the dependencies with pip:

```bash
pip install -r requirements.txt
```

Alternatively, create the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate vfstance
```

For API-based experiments, create `.env` in the repository root:

```text
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
```

## Dataset

K-News-Stance-MM is not included in this repository. Follow [`data/instructions.md`](data/instructions.md) to request the dataset, then place it at:

```text
data/k-news-stance-mm.json
```

Each record follows this structure:

```json
{
  "id": 42,
  "issue": "'개 식용 금지법' 국회 본회의 통과",
  "genre": "analysis",
  "news_outlet": "...",
  "link": "...",
  "publish_date": "...",
  "headline": "...",
  "images": [
    { "url": "...", "caption": "..." },
    { "url": "...", "caption": "..." }
  ],
  "article": "...",
  "article_stance": "neutral",
  "split": "test"
}
```

`article_stance` is one of `supportive`, `neutral`, or `oppositional`. Original news images are not distributed; only their source URLs are provided in `images[].url`. VFStance and VFStance (Text) do not require the original images. Visual and multimodal baselines require users to download the relevant images as described in the dataset instructions.

The dataset is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Running VFStance

Run all commands from the repository root. VFStance consists of visual-framing annotation, image generation, and multimodal stance prediction:

```bash
# Stage 1: article -> visual-framing annotation
python vfstance/stage1_annotate_visual_framing.py --split test

# Stage 2: annotation -> generated image
python vfstance/stage2_generate_image.py

# Stage 3: article + generated image -> stance
python vfstance/stage3_predict_stance.py --input-mode multimodal
```

Outputs are written under `vfstance/outputs/`.

### VFStance (Text)

VFStance (Text) skips image generation and supplies the visual description directly to the stance predictor. Run Stage 1 followed by:

```bash
python vfstance/stage3_predict_stance.py --input-mode text
```

Use `--help` on any script to view its paths, model settings, and asynchronous Batch API options.

## Open-Weight Experiments

The appendix's fully open configuration uses InternVL3-14B-Instruct for visual-framing annotation and stance prediction, and Stable Diffusion 3.5 Large for image generation. Install the additional dependencies and run:

```bash
pip install -r requirements-open.txt
python vfstance/run_open.py --stage all
```

## Baselines and Evaluation

```bash
# LVLM baselines
python baseline/lvlm/predict.py --help

# Evaluation
python evaluation/metrics.py path/to/results_dir

# EAIG4SD
cd eaig4sd
CONDA_ENV_NAME=myenv NUM_SHARDS=1 bash run_all.sh
```

## License & Attribution

The code is shared under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

Citation and attribution information will be added upon publication.
