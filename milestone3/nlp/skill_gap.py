import spacy
from nlp.bert_similarity import compute_similarity

nlp = spacy.load("en_core_web_sm")

TECH_SKILLS = [
    "python","sql","machine learning","deep learning",
    "tensorflow","pytorch","nlp","data analysis",
    "statistics","aws","cloud","excel"
]

SOFT_SKILLS = [
    "communication","teamwork","leadership",
    "problem solving","time management"
]

def extract_skills(text):
    text = text.lower()
    skills = []

    for skill in TECH_SKILLS + SOFT_SKILLS:
        if skill in text:
            skills.append(skill)

    return list(set(skills))

def skill_gap_analysis(resume_text, jd_text):
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    similarity_matrix = compute_similarity(resume_skills, jd_skills)

    matched = []
    partial = []
    missing = []

    for i, r_skill in enumerate(resume_skills):
        max_score = max(similarity_matrix[i])

        if max_score >= 0.8:
            matched.append(r_skill)
        elif max_score >= 0.5:
            partial.append(r_skill)

    for jd_skill in jd_skills:
        if jd_skill not in matched and jd_skill not in partial:
            missing.append(jd_skill)

    overall_match = int((len(matched) / len(jd_skills)) * 100) if jd_skills else 0

    return {
    "matched": matched,
    "partial": partial,
    "missing": missing,
    "overall_match": overall_match,

    
    "matched_count": len(matched),
    "partial_count": len(partial),
    "missing_count": len(missing),

    "resume_skills": resume_skills,
    "jd_skills": jd_skills,
    "similarity_matrix": similarity_matrix
}

