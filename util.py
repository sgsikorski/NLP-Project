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
