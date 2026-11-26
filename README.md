# VFStance

Official implementation of **Visual Framing for News Stance Detection via Image Generation** (EMNLP 2026) [[paper](https://aclanthology.org/2026.emnlp-main.2524/)].

VFStance leverages visual framing to make implicit stance cues in news articles explicit through image generation. It consists of three stages:

1. **Visual Framing Annotation**: an LLM annotates a visual framing specification for the article.
2. **Stance-aware Image Generation**: a text-to-image model generates an image from the specification.
3. **Stance Detection**: an LVLM predicts the article's stance from the article and the generated image.

<p align="center">
  <img src="assets/overview.png" width="100%">
</p>

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

For API-based models, export the API key or put it in `.env` in the repository root:

```text
GOOGLE_API_KEY=...
```

## Dataset

K-News-Stance-MM is not included in this repository. Follow [`data/README.md`](data/README.md) to request the dataset, then place it at:

```text
data/k_news_stance_mm/ko/k-news-stance-mm.json          # original Korean release
data/k_news_stance_mm/translated/{en,zh,id,ar}.json     # translated test split
```

Each record provides `issue`, `headline`, `article`, and a `stance` label that is one of `supportive`, `neutral`, or `oppositional`; see [`data/README.md`](data/README.md) for the full schema, the split statistics, an example record, and the translations. VFStance does not use the original news images; only their source URLs are provided in `images[].url`.

## Running VFStance

Run all commands from the repository root. Each stage reads the outputs of the previous stage and writes its own outputs under `outputs/`.

```bash
# Stage 1: article -> visual framing annotation
bash scripts/stage1_visual_framing_annotation.sh --backend api

# Stage 2: annotation -> generated image
bash scripts/stage2_stance_aware_image_generation.sh --backend api

# Stage 3: article + generated image -> stance
bash scripts/stage3_stance_detection.sh --backend api
```

To run all stages at once:

```bash
bash scripts/run_all.sh --config full --backend api
```

Use `--limit N` to run on the first N articles, and `--help` on any script to view its paths and model settings.

### VFStance (Text)

VFStance (Text) skips image generation and provides the Stage 1 annotation to the stance predictor as text, rendered through the same Stage 2 template:

```bash
bash scripts/run_all.sh --config text --backend api
```

It writes `outputs/predictions_text.jsonl`, so the two configurations do not overwrite each other. To run that stage alone, use `--input-mode text`.

### Open-Weight Models

The fully open configuration in the appendix uses InternVL3-14B for Stages 1 and 3 and Stable Diffusion 3.5 Large for Stage 2. Install the additional dependencies, log in to Hugging Face to access gated models, and run with `--backend open`:

```bash
pip install -r requirements-open.txt
huggingface-cli login
bash scripts/run_all.sh --config full --backend open
```

### Models

| Stage | `--backend api` | `--backend open` |
| --- | --- | --- |
| 1, 3 — LLM / LVLM | `gemini-3-flash-preview` | `OpenGVLab/InternVL3-14B-Instruct` |
| 2 — text-to-image | `gemini-3.1-flash-image-preview` | `stabilityai/stable-diffusion-3.5-large` |

These are only defaults: any model can be swapped in on either backend with `--model` on an individual stage, or `--lvlm-model` / `--t2i-model` on `run_all.sh`. For API models, providers other than Gemini plug into [`src/vfstance/backends/api.py`](src/vfstance/backends/api.py) through `PROVIDER_PREFIXES`.

## Additional Materials

- [`docs/prompts_translated.md`](docs/prompts_translated.md): English translations of prompts not translated in the paper
- [`docs/case_studies.md`](docs/case_studies.md): the nine articles used in the case study, and case studies not included in the paper

## License

- Data: To respect the intellectual property rights of the original news publishers, K-News-Stance-MM is released under gated access with a custom Data Use Agreement: prospective users must submit an access request describing their research purpose and agree to terms that restrict use to non-commercial academic research and prohibit redistribution.
- Code: The code is shared under the MIT License. 

## Citation

```bibtex
@inproceedings{lee-etal-2026-visual,
    title = "Visual Framing for News Stance Detection via Image Generation",
    author = "Lee, Dahyun  and
      Han, Jiyoung  and
      Park, Kunwoo",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Junyi Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.2524/",
    doi = "10.18653/2026.emnlp-main.2524",
    pages = "57393--57418",
    ISBN = "979-8-89176-459-0",
    abstract = "Article-level news stance detection aims to identify the perspective of news articles toward social issues. Despite advances in stance detection and its importance for trustworthy media environments, news articles pose distinct challenges because their stances are often implicit, subtly conveyed through journalistic framing, and embedded in long, structurally complex texts. To address these challenges, we introduce VFStance, which leverages visual framing to make implicit stance cues more explicit via image generation. In evaluation experiments, we demonstrate the effectiveness of VFStance over existing methods and the contribution of visual framing to its performance. Finally, a controlled user study (N=200) in a snippet-based news consumption setting further demonstrates that VFStance can make stance signals visually salient and highlights its potential use beyond automated stance detection."
}
```
