import numpy as np
from transformers import (
    pipeline,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments
from collections import defaultdict
import torch
import logging

logger = logging.getLogger(__name__)

def getPrompt(note, entities):
    prompt = f"""
You are an expert medical language model tasked with analyzing clinical notes to determine patient recovery outcomes. Given a clinical note and extracted entities, assess the sentiment of the note with respect to the patient's recovery risk.
Clinical Note: {note}
Entities: {entities}
Assess the sentiment of the clinical note with respect to the patient's recovery risk.
Positive: Indicators of improvement or a high likelihood of recovery.
Neutral: Indicators of stability or uncertain outcomes.
Negative: Indicators of deterioration or a low likelihood of recovery.
"""
    return prompt


def finetune():
    torch.set_printoptions(profile="full")
    torch.autograd.set_detect_anomaly(True)

    from datasets import load_dataset

    data = load_dataset("zhengyun21/PMC-Patients")

    logger.info("Loaded Dataset")

    from generateSentiment import generate_sentiment

    data["train"] = data["train"].select(range(10))

    # Add a preliminary sentiment label to the dataset
    sentiments = generate_sentiment(data["train"])
    sentimentMap = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    sentiments = [sentimentMap[sentiment["label"]] for sentiment in sentiments]
    data["train"] = data["train"].add_column("label", sentiments)

    logger.info(f"Add {len(sentiments)} pretrained sentiments")
    from findEntities import find_entities

    entities = find_entities(data["train"])
    # data["train"] = data["train"].add_column("entities", entities)

    from huggingface_hub import HfFolder

    token = HfFolder.get_token()

    pretrained_model_name = "chaoyi-wu/PMC_LLAMA_7B"
    llama_model_name = "meta-llama/Llama-3.2-1B"
    model_name = llama_model_name if token else pretrained_model_name
    # model_name = pretrained_model_name
    # pipe = pipeline(
    #     "text-generation",
    #     model=model_name,
    #     device=-1
    # )

    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3,
        quantization_config=quant_config
    )# .to("cuda:0")

    llm_model = prepare_model_for_kbit_training(llm_model)
    lora_config = LoraConfig(
        r = 8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none"
    )
    llm_model = get_peft_model(llm_model, lora_config)

    if llm_tokenizer.pad_token is None:
        llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        llm_model.resize_token_embeddings(len(llm_tokenizer))
        logger.info("Add padding token to tokenizer")
    llm_tokenizer.pad_token = '[PAD]'
    if llm_model.config.pad_token_id is None:
        logger.info("Add padding token to model")
        llm_model.config.pad_token_id = llm_tokenizer.pad_token_id

    prompts = {"input_ids": [], "attention_mask": []}
    for note, entity_entry in zip(data["train"], entities):
        result = llm_tokenizer(getPrompt(note["patient"], entity_entry), padding="max_length", truncation=True, max_length=256)
        prompts["input_ids"].append(result["input_ids"])
        prompts["attention_mask"].append(result["attention_mask"])

    data["train"] = data["train"].add_column("input_ids", prompts["input_ids"])
    data["train"] = data["train"].add_column("attention_mask", prompts["attention_mask"])

    del prompts["input_ids"]
    del prompts["attention_mask"]
    torch.cuda.empty_cache() 

    logger.info(f"Add {len(entities)} pretrained entities")

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

    logger.info("Split the dataset")

    logger.info(f"Loaded {model_name}")

    sentiment_training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        # fp16=True,
    )

    sentiment_trainer = Trainer(
        model=llm_model,
        args=sentiment_training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["validation"],
        tokenizer=llm_tokenizer,
    )

    logger.log(logging.CRITICAL, "Starting finetuning")

    sentiment_trainer.train()
    logger.info("Finetuning done!")
    llm_model.save_pretrained("./results")
    llm_tokenizer.save_pretrained("./results")
    logger.info("Model and tokenizer saved to results")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    logging.basicConfig(filename='finetune.log', level=logging.INFO, filemode='w')
    finetune()
