import spacy
from .skills_db import TECHNICAL_SKILLS, SOFT_SKILLS

nlp = spacy.load("en_core_web_sm")

def extract_skills_spacy(text):
    doc = nlp(text.lower())
    tech, soft = set(), set()

    for token in doc:
        if token.text in TECHNICAL_SKILLS:
            tech.add(token.text)
        if token.text in SOFT_SKILLS:
            soft.add(token.text)

    return {
        "technical": list(tech),
        "soft": list(soft)
    }
