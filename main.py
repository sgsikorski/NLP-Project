import numpy as np
from transformers import pipeline
from collections import defaultdict
import torch

from datasets import load_dataset

data = load_dataset("zhengyun21/PMC-Patients")

from huggingface_hub import HfFolder

token = HfFolder.get_token()

pretrained_model_name = "chaoyi-wu/PMC_LLAMA_7B"
llama_model_name = "meta-llama/Llama-3.2-1B"
model_name = llama_model_name if token else pretrained_model_name
# model_name = pretrained_model_name
pipe = pipeline(
    "text-generation", model=model_name, device=0 if torch.cuda.is_available() else -1
)


# Extract medical entities from the clinical note
# Classify them into conditions, treatments, and outcomes
def named_entity_recognition(title, note):
    prompt = f"""
Clinical Note: {note}
You are a specialized medical language model designed to extract critical information from clinical notes. Given the clinical note, identify and extract the following entities:

Symptoms: Physical or psychological conditions reported by the patient.
Diagnoses: Medical conditions or diseases identified by the clinician.
Treatments/Medications: Procedures, therapies, or drugs mentioned.
Outcomes: Observations or indications of the patient's response to treatment or prognosis
Return the extracted entities categorized into the corresponding groups. Use accurate medical terminology, and only include entities explicitly or implicitly mentioned in the text.
Example Input:
"Patient reports severe fatigue and joint pain. Diagnosed with rheumatoid arthritis. Prescribed methotrexate. Follow-up shows improved joint mobility but persistent mild fatigue. Recent ESR levels have decreased but are still elevated."
Expected Output:
Entities:
Symptoms: severe fatigue, joint pain
Diagnoses: rheumatoid arthritis
Treatments/Medications: methotrexate
Outcomes: improved joint mobility, persistent mild fatigue
    """
    result = pipe(prompt, max_new_tokens=25, num_return_sequences=1)
    print(result)
    entities = result[0]["generated_text"].split("Entities:")[-1].strip()
    return entities


# Determine the patient's recovery risk sentiment
# Positive (low risk), neutral (medium risk), or negative (high risk)
def sentiment_analysis(note, entities):
    prompt = f"""
You are an expert medical language model tasked with analyzing clinical notes to determine patient recovery outcomes. Given a clinical note and extracted entities, assess the sentiment of the note with respect to the patient's recovery risk.
Clinical Note: {note}
Entities: {entities}
Assess the sentiment of the clinical note with respect to the patient's recovery risk.
Positive: Indicators of improvement or a high likelihood of recovery.
Neutral: Indicators of stability or uncertain outcomes.
Negative: Indicators of deterioration or a low likelihood of recovery.
Example Input:
"Patient presents with severe dyspnea and elevated BNP levels. Treatment initiated with diuretics shows mild improvement. However, recurring chest pain persists, and cardiac markers remain elevated."
Expected Output:
Sentiment: Neutral
    """
    result = pipe(prompt, max_new_tokens=10, num_return_sequences=1)
    print(result)
    return result[0]["generated_text"].split("Sentiment:")[-1].strip()


labeled_notes = []
for i, entry in enumerate(data["train"]):
    title = entry["title"]
    note = entry["patient"]
    entities = named_entity_recognition(title, note)
    sentiment = sentiment_analysis(note, entities)
    labeled_notes.append(
        {"title": title, "note": note, "entities": entities, "sentiment": sentiment}
    )
    break
print(labeled_notes[0]["entities"])
print(labeled_notes[0]["sentiment"])

entity_sentiment_map = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})

for entry in labeled_notes:
    sentiment = entry["sentiment"]
    entities = entry["entities"]
    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            entity_sentiment_map[entity][sentiment] += 1
