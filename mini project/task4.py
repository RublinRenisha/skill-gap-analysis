import re
def normalize_job_title(text):
    cleaned = re.sub(r'[^A-Za-z\s]', '', text)
    normalized = cleaned.title()
    return normalized
job_title = "Entry-Level Data Scientist"
normalized_title = normalize_job_title(job_title)
print("Normalized job title:", normalized_title)
