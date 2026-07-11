import json
import os
import random
import re
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from gensim import corpora
from gensim.models.ldamodel import LdaModel
 
nltk.download("stopwords")


DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "k-news-stance-mm-acl-arr.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "event_topic_words.json"


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed_value}")

set_seed(42)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

with DATA_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
train_df = df[df["split"] == "train"]

def tokenize_korean(text):
    text = re.sub(r"[^ㄱ-ㅎ가-힣\s]", "", text)
    return text.split()

grouped = train_df.groupby("issue")["article"].apply(lambda docs: " ".join(docs)).to_dict()
event_topic_words = {}

for event, corpus_text in grouped.items():
    tokens = [tokenize_korean(sent) for sent in corpus_text.split("\n") if sent.strip()]
    tokens = [t for t in tokens if t]

    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(t) for t in tokens]

    if len(dictionary) < 10:
        continue

    lda = LdaModel(
        corpus, 
        num_topics=1, 
        id2word=dictionary, 
        passes=10,
        random_state=42  
    )

    top_words = lda.show_topic(0, topn=10)
    topic_words = [word for word, prob in top_words]
    event_topic_words[event] = topic_words

for e, words in list(event_topic_words.items()):
    print(f"{e} → {words}")

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(event_topic_words, f, ensure_ascii=False, indent=2)
