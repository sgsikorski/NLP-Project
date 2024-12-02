from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("Junaidb/medicalNER")
model = AutoModelForTokenClassification.from_pretrained("Junaidb/medicalNER")


# TODO: Filter out which entities are necessary
def getEntity(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=2)
    label_map = model.config.id2label
    entities = []
    for token, prediction in zip(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), predictions[0]
    ):
        entity_label = label_map[prediction.item()]
        if token.startswith("##"):
            entities[-1] = (entities[-1][0], entities[-1][1] + token[2:])
        else:
            if entity_label != "O":
                if "_" in entity_label:
                    entity_label = entity_label[1:]
                entities.append((entity_label, token))
    return entities


def find_entities(data):
    entities = []
    for entry in data:
        entities.append(getEntity(entry["patient"]))
    return entities


if __name__ == "__main__":
    data = load_dataset("zhengyun21/PMC-Patients")
    entities = find_entities(data["train"])
    print(entities)
