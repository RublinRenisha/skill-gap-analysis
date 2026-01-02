def merge_skills(spacy_skills, bert_skills):
    tech = set(spacy_skills["technical"] + bert_skills["technical"])
    soft = set(spacy_skills["soft"] + bert_skills["soft"])

    return {
        "technical": list(tech),
        "soft": list(soft),
        "tech_count": len(tech),
        "soft_count": len(soft),
        "total": len(tech) + len(soft)
    }
