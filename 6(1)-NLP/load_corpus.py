# 구현하세요!
from datasets import load_dataset

ds = load_dataset("google-research-datasets/poem_sentiment")

def load_corpus() -> list[str]:
    corpus: list[str] = []
    # 구현하세요!
    for item in ds["train"]:
        corpus.append(item["verse_text"])
    return corpus