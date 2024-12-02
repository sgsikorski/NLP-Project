import numpy as np
from transformers import (
    pipeline,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers import Trainer, TrainingArguments
from collections import defaultdict
import torch


def finetune():
    from datasets import load_dataset

    data = load_dataset("zhengyun21/PMC-Patients")

    from generateSentiment import generate_sentiment

    data["train"] = data["train"].select(range(10))

    # Add a preliminary sentiment label to the dataset
    sentiments = generate_sentiment(data["train"])
    sentimentMap = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    sentiments = [sentimentMap[sentiment["label"]] for sentiment in sentiments]
    data["train"] = data["train"].add_column("sentiment_label", sentiments)

    from findEntities import find_entities

    entities = find_entities(data["train"])
    data["train"] = data["train"].add_column("entities", entities)

    train_split = 0.8
    val_split = 0.1
    test_split = 1 - train_split - val_split
    assert train_split + val_split + test_split == 1

    data = data["train"].train_test_split(test_size=0.2, seed=0)
    test_val_split = data["test"].train_test_split(test_size=0.5, seed=0)
    # Combine splits into a single dataset
    split_dataset = {
        "train": data["train"],
        "validation": test_val_split["train"],
        "test": test_val_split["test"],
    }

    from huggingface_hub import HfFolder

    token = HfFolder.get_token()

    pretrained_model_name = "chaoyi-wu/PMC_LLAMA_7B"
    llama_model_name = "meta-llama/Llama-3.2-1B"
    model_name = llama_model_name if token else pretrained_model_name
    # model_name = pretrained_model_name
    pipe = pipeline(
        "text-generation",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )

    sentiment_training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
    )

    sentiment_trainer = Trainer(
        model=sentiment_model,
        args=sentiment_training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["validation"],
        processing_class=tokenizer.__class__,
    )

    sentiment_trainer.train()
    sentiment_model.save_pretrained("./trained_sentiment_model")
    tokenizer.save_pretrained("./trained_sentiment_tokenizer")


if __name__ == "__main__":
    finetune()
