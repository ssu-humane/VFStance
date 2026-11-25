# Prompts and Translations

The prompt files the code actually reads live in [`../prompts/`](../prompts/) and
are written in English:

| File | Used by |
| --- | --- |
| [`stage1_visual_framing_annotation.txt`](../prompts/stage1_visual_framing_annotation.txt) | Stage 1 |
| [`stage2_stance_aware_image_generation_template.txt`](../prompts/stage2_stance_aware_image_generation_template.txt) | Stage 2 |
| [`stage3_stance_detection.txt`](../prompts/stage3_stance_detection.txt) | Stage 3, VFStance |
| [`stage3_stance_detection_text.txt`](../prompts/stage3_stance_detection_text.txt) | Stage 3, VFStance (Text) |

Stage 1 instructs the annotator to write the visual framing spec in Korean, the
language of the articles, so the slots filled into the Stage 2 template are
Korean even though the template itself is English. This page gives English
translations of that filled-in content, which the paper shows only in Korean.

## Stage 2: text-to-image prompt

### Original (Korean), as reported in the paper

Slot values filled from the Stage 1 spec are *italicized*.

> An *illustration*-style image.
> Include the following: *텅 빈 지방 도시들의 실루엣, 거대한 서울 중심의 블랙홀, 선거용 홍보 문구가 적힌 투표함, 어지럽게 뒤섞인 경기도 행정 구역 지도, 김포, 구리, 하남, 고양, 부천, 광명 지명*.
> Composition is *asymmetric*, *high* angle and *long* shot. Saturation is *desaturated*, luminosity is *dark*.
> Exclude the following: *조경태 의원, 이재명 대표, 밝게 웃는 시민들, 화려한 서울의 야경*.
> Generate only one image at a time.

### English translation

> An *illustration*-style image.
> Include the following: *silhouettes of emptied provincial cities, a giant black hole centered on Seoul, a ballot box covered in campaign slogans, a chaotically scrambled map of the administrative districts of Gyeonggi Province, the place names Gimpo, Guri, Hanam, Goyang, Bucheon, and Gwangmyeong*.
> Composition is *asymmetric*, *high* angle and *long* shot. Saturation is *desaturated*, luminosity is *dark*.
> Exclude the following: *Representative Cho Kyoung-tae, Party Leader Lee Jae-myung, brightly smiling citizens, the glittering night view of Seoul*.
> Generate only one image at a time.

The same filled-in prompt is what VFStance (Text) passes to Stage 3 in place of
an image, through the `{visual_description}` slot of
[`stage3_stance_detection_text.txt`](../prompts/stage3_stance_detection_text.txt).
