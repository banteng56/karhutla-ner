"""Generate reproducible silver NER labels for the karhutla news corpus."""

import json
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline


ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
INPUT_FILE = DATASETS / "indonesia_fire_news_2y_deduplicated.csv"
OUTPUT_CSV = DATASETS / "ner_silver_labeled.csv"
OUTPUT_JSONL = DATASETS / "ner_silver_labeled.jsonl"
MODEL_NAME = "cahya/bert-base-indonesian-NER"
BATCH_SIZE = 32
MIN_SCORE = 0.75

LABEL_MAP = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
    "GPE": "LOC",
    "DAT": "DATE",
    "TIM": "TIME",
    "EVT": "EVENT",
}


def normalize_predictions(text, predictions):
    entities = []
    for item in predictions:
        score = float(item["score"])
        if score < MIN_SCORE:
            continue
        start, end = item["start"], item["end"]
        entities.append(
            {
                "start": start,
                "end": end,
                "label": LABEL_MAP.get(item["entity_group"], item["entity_group"]),
                "model_label": item["entity_group"],
                "text": text[start:end],
                "score": round(score, 4),
            }
        )
    return entities


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    news = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False)

    summary = news["summary"].astype(str)
    excerpt = news["content_excerpt"].astype(str)
    silver = news[["id", "title"]].copy()
    silver["text"] = (
        news["title"].astype(str)
        + ". "
        + summary.where(summary.str.strip().ne(""), excerpt)
    )
    silver["text"] = (
        silver["text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.slice(0, 450)
    )

    ner = pipeline(
        "token-classification",
        model=MODEL_NAME,
        aggregation_strategy="simple",
        device=-1,
    )
    ner.tokenizer.model_max_length = 512

    all_predictions = []
    texts = silver["text"].tolist()
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        all_predictions.extend(ner(batch, batch_size=BATCH_SIZE))
        completed = min(start + BATCH_SIZE, len(texts))
        print(f"Auto-label: {completed:,}/{len(texts):,}", flush=True)

    silver["entities"] = [
        normalize_predictions(text, prediction)
        for text, prediction in zip(texts, all_predictions)
    ]
    silver["entity_count"] = silver["entities"].str.len()
    silver["entities_json"] = silver["entities"].apply(json.dumps, ensure_ascii=False)
    silver["label_source"] = "silver:cahya/bert-base-indonesian-NER"

    silver.drop(columns="entities").to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    with OUTPUT_JSONL.open("w", encoding="utf-8") as file:
        for row in silver.itertuples(index=False):
            file.write(
                json.dumps(
                    {
                        "id": row.id,
                        "text": row.text,
                        "entities": row.entities,
                        "label_source": row.label_source,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Saved {len(silver):,} labeled documents to {OUTPUT_CSV}")
    print(f"Saved annotation JSONL to {OUTPUT_JSONL}")
    print(f"Total entities: {silver['entity_count'].sum():,}")


if __name__ == "__main__":
    main()
