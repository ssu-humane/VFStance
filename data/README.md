# K-News-Stance-MM

K-News-Stance-MM is the multimodal Korean news stance detection dataset used in
*Visual Framing for News Stance Detection via Image Generation*. Each record is a
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

### Example record

**Target issue** — ‘양곡관리법 개정안’ 민주당 주도 본회의 통과

**Headline** — 李 ‘1호 법안’ 양곡법 尹 ‘1호 거부권’ 유력

**Article**

> 정부가 초과 생산된 쌀을 의무적으로 매입하는 내용을 담은 양곡관리법 개정안이 23일 국회에서 더불어민주당과 정의당 등 야당 주도로 본회의를 통과했다. 윤석열 대통령은 양곡관리법 개정안에 대해 ‘거부권’(재의요구권)을 행사하는 방안을 검토 중인 것으로 알려졌다. 이에 따라 이재명 민주당 대표가 ‘1호 법안’으로 선정했던 양곡관리법이 윤 대통령의 ‘1호 거부권’ 행사 법안이 될 가능성이 커졌다. 윤 대통령이 실제로 거부권을 행사할 경우 2016년 5월 박근혜 당시 대통령이 국회 상임위의 ‘상시 청문회’ 개최를 핵심으로 하는 국회법 개정안에 거부권을 행사한 이후 약 7년 만의 거부권 행사가 될 전망이다. 양곡관리법 개정안은 이날 국회 본회의에서 재석 266명 중 찬성 169명, 반대 90명, 기권 7명으로 가결됐다. 개정안은 쌀 수요 대비 초과 생산량이 3~5% 이상이거나 쌀값이 전년 대비 5~8% 하락할 때 정부가 초과 생산량을 전량 매입하는 것을 핵심 내용으로 한다. 민주당은 쌀값 안정화와 농민 보호, 식량 주권 수호 등을 앞세워 양곡관리법을 밀어붙였다. 그러나 정부와 국민의힘은 쌀의 공급과잉과 정부 재정부담, 농업경쟁력 약화 등을 이유로 반대했으나 수적 열세를 극복하지 못했다. 정부는 개정안 처리에 깊은 유감을 표했다. 정황근 농림축산식품부 장관은 정부서울청사에서 열린 긴급 브리핑에서 “부작용이 너무나 명백하다”며 수용 불가 입장을 밝혔다. 야당의 강행 처리로 개정안이 국회를 통과하면서 법안 부작용에 대한 우려가 커지고 있다. 개정안 통과로 정부는 연평균 20만t 안팎의 쌀을 더 사들일 것으로 예상되는데 이로 인한 비용이 약 1조원 정도로 추산된다. 국민 식습관 변화로 쌀에 대한 수요가 떨어지고 있어 의무 수매한 쌀은 정부 창고에 쌓여있다가 막걸리 제조 등 주정용으로 헐값에 팔릴 가능성이 크다는 주장도 있다. 또 정부의 의무매입 확대로 인한 쌀 생산량 확대 그리고 이로 인한 쌀값 하락이 고착화할 수 있다는 우려도 크다. 윤 대통령은 양곡관리법 개정안에 대한 거부권 행사에 무게를 두고 숙고할 것으로 알려졌다. 대통령실 고위 관계자는 “개정안이 다음 주 정부로 이송되면 여론 수렴과 고심의 시간을 가질 것”이라고 밝혔다.

**Image** — <https://imgnews.pstatic.net/image/005/2023/03/24/2023032321490368287_1679575743_0924293423_20230324041205219.jpg?type=w860>
(the image file itself is not redistributed; only the URL is provided)

**Stance** — `oppositional`

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
translation (Gemini Batch API) of the `issue`, `headline`, and `article` fields;
`id`, `stance`, `split`, and all metadata are carried over unchanged from the
Korean release, so records can be joined across languages by `id`. Each record
additionally carries a `language` field (`en`, `zh`, `id`, `ar`).

| Language   | File                 |
| ---------- | -------------------- |
| English    | `translated/en.json` |
| Chinese    | `translated/zh.json` |
| Indonesian | `translated/id.json` |
| Arabic     | `translated/ar.json` |

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
