from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
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
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    **model_args,
)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    **model_args,
)


def chunk_text(text, max_length=512):
    tokens = tokenizer(text, truncation=False)["input_ids"]
    chunks = [tokens[i : i + max_length] for i in range(0, len(tokens), max_length)]
    return chunks


def classify_chunks(chunks):
    scores = []
    for chunk in chunks:
        inputs = tokenizer.decode(chunk)
        result = classifier(inputs)
        scores.append(
            result[0]["score"]
            if result[0]["label"] == "LABEL_1"
            else -result[0]["score"]
        )
    print(scores)
    return np.mean(scores)


def generate_sentiment(entries):
    labelMap = {"LABEL_0": "Negative", "LABEL_1": "Positive"}
    labels = []
    for i, entry in enumerate(entries):
        chunks = chunk_text(entry["patient"])
        average_score = classify_chunks(chunks)
        label = "LABEL_1" if average_score > 0 else "LABEL_0"
        labels.append({"label": labelMap[label], "score": average_score})
    return labels


if __name__ == "__main__":
    data = load_dataset("zhengyun21/PMC-Patients")
    results = generate_sentiment(data["train"])
    print(results)
