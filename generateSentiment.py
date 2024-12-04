from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    pipeline,
)
from datasets import load_dataset
import numpy as np

model_args = {
    "max_length": 512,
    "truncation": True,
    "padding": "max_length",
}

model_name = "emilyalsentzer/Bio_ClinicalBERT"
_tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    **model_args,
)

_model = AutoModelForMaskedLM.from_pretrained(model_name)

_classifier = pipeline(
    "fill-mask",
    model=_model,
    tokenizer=_tokenizer,
)


def chunk_text(text, max_length=512, tokenizer=None):
    if tokenizer is None:
        tokenizer = _tokenizer
    tokens = tokenizer(text, truncation=False)["input_ids"]
    chunks = [
        tokens[i : i + max_length - 25] for i in range(0, len(tokens), max_length)
    ]
    return chunks


def classify_chunks(chunks, tokenizer=None, classifier=None):
    if tokenizer is None:
        tokenizer = _tokenizer
    if classifier is None:
        classifier = _classifier
    scores = []
    for chunk in chunks:
        chunk += tokenizer(
            "Of high, medium, or low, the patient's recovery risk is [MASK]",
            padding=False,
        )["input_ids"]
        inputs = tokenizer.decode(chunk)
        result = classifier(inputs)
        label = tokenizer.decode(result[0]["token"])
        bad = [
            "high",
            "likely",
            "concerning",
            "increased",
            "significantly",
            "notable",
            "limited",
        ]
        good = ["low", "negative", "0", "not"]
        neutral = [
            "medium",
            "unclear",
            "?",
            "neutral",
            "unknown",
            "normal",
            "consistent",
        ]
        if label in bad:
            scores.append(-result[0]["score"])
        elif label in neutral:
            scores.append(0)
        elif label in good:
            scores.append(result[0]["score"])
        else:
            scores.append(0)
    return np.mean(scores)


def generate_sentiment(entries):
    labels = []
    for i, entry in enumerate(entries):
        chunks = chunk_text(entry["patient"])
        average_score = classify_chunks(chunks)
        label = ""
        if average_score > 0.1:
            label = "POSITIVE"
        elif average_score < -0.1:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        labels.append({"label": label, "score": average_score})
    return labels


if __name__ == "__main__":
    # For testing purposes
    data = load_dataset("zhengyun21/PMC-Patients")
    data["train"] = data["train"].select(range(10))
    results = generate_sentiment(data["train"])
    print(results)
