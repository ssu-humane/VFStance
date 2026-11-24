# K-News-Stance-MM

K-News-Stance-MM is the multimodal Korean news stance detection dataset used in
*Visual Framing for News Stance Detection via Generated Images*. Each record is a
news article paired with a target issue, the article's stance toward that issue,
and the images that the article was published with.

## Access

The dataset is not distributed directly in this repository.

1. Open the **Download Form** (the URL will be added upon publication).
2. Submit the requested information.
3. A download link is sent to your email.
4. Place the files as follows:

```text
data/k_news_stance_mm/
├── ko/
│   └── k-news-stance-mm.json          # original Korean release
└── translated/
    ├── en.json                        # English
    ├── zh.json                        # Chinese
    ├── id.json                        # Indonesian
    └── ar.json                        # Arabic
```

VFStance and VFStance (Text) need only these JSON files. Original news images are
not redistributed; see [Original news images](#original-news-images) below.

## Source

Articles were collected from Naver News, the main news portal in South Korea,
covering 47 socially contested issues reported by 31 press outlets between
June 2022 and June 2024. Every article is annotated with its stance toward the
issue it covers.

## Schema

```json
{
  "id": 728,
  "issue": "헌재 '검수완박' 절차는 위헌, 법안은 유효",
  "headline": "“검수완박, 檢권한 침해 아냐…법무장관 청구인 자격 없다”",
  "article": "헌재, 법무부 등 무효확인 청구 각하 ...",
  "press": "국제신문",
  "date": "20230324",
  "naver_link": "https://n.news.naver.com/mnews/article/658/0000036368?sid=100",
  "images": [
    { "url": "https://imgnews.pstatic.net/image/658/...jpg", "caption": "..." }
  ],
  "stance": "neutral",
  "genre": "analysis",
  "split": "test"
}
```

| Field | Type | Description |
| --- | --- | --- |
| `id` | int | Unique article id. Used in every pipeline output and in the generated image filenames. |
| `issue` | string | Target issue the stance is annotated against. |
| `headline` | string | Article headline. |
| `article` | string | Article body text. |
| `press` | string | Publishing outlet. |
| `date` | string | Publication date (`YYYYMMDD`). |
| `naver_link` | string | Source URL on Naver News. |
| `images` | list | Images published with the article: `url` and `caption`. One image per article. |
| `stance` | string | Gold label: `supportive`, `neutral`, or `oppositional`. |
| `genre` | string | `analysis` (1,653 articles) or `opinion` (163 articles). |
| `split` | string | `train`, `validation`, or `test`. |

The stance label is the article's stance toward `issue`, not toward a generic
target. VFStance uses `issue`, `headline`, and `article` as input in Stage 1 and
Stage 3, and never uses `stance` as model input.

## Splits

| Split | Supportive | Neutral | Oppositional | Total |
| --- | --- | --- | --- | --- |
| train | 223 | 248 | 258 | 729 |
| validation | 56 | 62 | 62 | 180 |
| test | 289 | 305 | 313 | 907 |
| **all** | **568** | **615** | **633** | **1,816** |

All experiments in the paper are zero-shot and report results on the `test`
split; `--split test` is the default in every stage script.

## Translations

The `translated/` releases cover the **test split** and were produced by machine
translation of the `issue`, `headline`, and `article` fields; `id`, `stance`,
`split`, and all metadata are carried over unchanged from the Korean release, so
records can be joined across languages by `id`.

| Language | File | Translated test articles |
| --- | --- | --- |
| English | `translated/en.json` | 880 / 907 |
| Chinese | `translated/zh.json` | 857 / 907 |
| Indonesian | `translated/id.json` | 873 / 907 |
| Arabic | `translated/ar.json` | 812 / 907 |

Articles whose translation could not be parsed are omitted, which is why the
counts differ per language. To run the pipeline on a translation, point every
stage at it:

```bash
bash scripts/run_all.sh --config full --backend open \
  --data-path data/k_news_stance_mm/translated/en.json
```

## Original news images

Original news image files are neither in this repository nor in the dataset
download; the dataset provides their source URLs in `images[].url`. VFStance does
not need them -- the pipeline generates its own images in Stage 2.

If you need the original images for your own comparison, download them and save
them under a local directory, for example:

```text
data/k_news_stance_mm/images/article_{id}_image_{index}.{ext}
```

where `index` is the zero-based position in the record's `images` list.

## License

K-News-Stance-MM is licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): share and adapt
for non-commercial purposes with attribution. The article text remains the
copyright of the respective publishers and is redistributed for non-commercial
research use only. See [`../LICENSE`](../LICENSE) for how this relates to the
code license. Download-form and citation details will be added upon publication.
