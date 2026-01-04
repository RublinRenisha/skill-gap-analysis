from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_similarity(resume_skills, jd_skills):
    if not resume_skills or not jd_skills:
        return []

    resume_embeddings = model.encode(resume_skills)
    jd_embeddings = model.encode(jd_skills)

    similarity = util.cos_sim(resume_embeddings, jd_embeddings)
    return similarity.tolist()
