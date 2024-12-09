import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from collections import defaultdict
import torch
import logging
from datasets import load_dataset
from huggingface_hub import HfFolder

from peft import PeftModel, PeftConfig

import argparse as ap

from util import getPrompt
from findEntities import getEntity
from finetune import finetune

logger = logging.getLogger(__name__)


def preprocessData(data):
    from generateSentiment import generate_sentiment

    data["train"] = data["train"].filter(lambda x: len(x["patient"]) <= 250)

    # Add a preliminary sentiment label to the dataset
    sentiments = generate_sentiment(data["train"])
    sentimentMap = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    sentiments = [sentimentMap[sentiment["label"]] for sentiment in sentiments]
    data["train"] = data["train"].add_column("label", sentiments)

    logger.info(f"Add {len(sentiments)} pretrained sentiments")
    return data


def main():
    logging.basicConfig(filename="finetune.log", level=logging.INFO, filemode="w")
    parser = ap.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    data = load_dataset("zhengyun21/PMC-Patients")
    data = preprocessData(data)

    if args.load:
        ft_model = AutoModelForSequenceClassification.from_pretrained(
            "meta-llama/Llama-3.2-1B"
        )
        ft_model = PeftModel.from_pretrained(ft_model, "./results")
        ft_tokenizer = AutoTokenizer.from_pretrained("./results")
    elif args.train:
        ft_model, ft_tokenizer = finetune(data, args.save)

    data = data["train"].train_test_split(test_size=0.2, seed=0)
    test_val_split = data["test"].train_test_split(test_size=0.5, seed=0)
    # Combine splits into a single dataset
    test_data = test_val_split["test"]

    token = HfFolder.get_token()
    pretrained_model_name = "chaoyi-wu/PMC_LLAMA_7B"
    llama_model_name = "meta-llama/Llama-3.2-1B"
    model_name = llama_model_name if token else pretrained_model_name
    baseline_pipe = pipeline(
        "text-generation",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
    )
    ft_pipe = pipeline(
        "sequence-classification", model=ft_model, tokenizer=ft_tokenizer
    )
    for entry in test_data:
        note = entry["patient"]
        entities = getEntity(note)
        prompt = getPrompt(note, entities)
        base_sentiment = baseline_pipe(prompt)[0]["generated_text"]
        ft_sentiment = ft_pipe(prompt)[0]["label"]

        logger.info(f"Base sentiment: {base_sentiment}")
        logger.info(f"Fine-tuned sentiment: {ft_sentiment}")
