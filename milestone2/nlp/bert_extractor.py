from transformers import pipeline
from .skills_db import TECHNICAL_SKILLS, SOFT_SKILLS

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    framework="pt"
)

def extract_skills_bert(text):
    labels = TECHNICAL_SKILLS + SOFT_SKILLS
    result = classifier(text, labels)

    tech, soft = [], []

    for label, score in zip(result["labels"], result["scores"]):
        if score > 0.70:
            if label in TECHNICAL_SKILLS:
                tech.append(label)
            else:
                soft.append(label)

    return {
        "technical": tech,
        "soft": soft
    }
