import spacy
nlp = spacy.load("en_core_web_sm")

from spacy.matcher import PhraseMatcher
text = "I have experience in Python and SQL."
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
skills = ["Python", "SQL"]
patterns = [nlp(skill) for skill in skills]
matcher.add("TECH_SKILLS", patterns)
doc = nlp(text)
found = []
for match_id, start, end in matcher(doc):
    found.append(doc[start:end].text.lower())
print(found)
#['python', 'sql']

tech_skills = ["Python", "SQL", "NLP", "Machine Learning", "Java"]
patterns = [nlp(skill) for skill in tech_skills]

lower_skills = [skill.lower() for skill in found]
print(lower_skills)
#['python', 'sql']

unique_skills = list(set(lower_skills))
print(unique_skills)
#['python', 'sql']

text = "Experience in Python, NLP, and Machine Learning with SQL."
doc = nlp(text)
matches = matcher(doc)
skills_found = list(set([doc[start:end].text.lower() for _, start, end in matches]))
print(skills_found)
#['python', 'sql']

soft_skills = ["communication", "teamwork", "leadership", "problem solving"]
text = "Good communication and teamwork skills."
doc = nlp(text)
soft_found = []
for token in doc:
    if token.text.lower() in soft_skills:
        soft_found.append(token.text.lower())
print(list(set(soft_found)))
#['communication', 'teamwork']

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

import json
output = {
    "technical_skills": skills_found,
    "soft_skills": soft_found
}
print(output)
with open("skills.json", "w") as f:
    json.dump(output, f, indent=4)

text = "Python, SQL, Python, NLP, SQL"
doc = nlp(text)
matches = matcher(doc)
skills = list(set([doc[start:end].text.lower() for _, start, end in matches]))
print(skills)
#[]

text = "Skills: Python, SQL; NLP; Machine Learning."
doc = nlp(text)
matches = matcher(doc)
skills = list(set([doc[start:end].text.lower() for _, start, end in matches]))
print(skills)
#[]

def extract_skills(text):
    doc = nlp(text)
    tech = list(set([doc[start:end].text.lower() for _, start, end in matcher(doc)]))
    soft = []
    for token in doc:
        if token.text.lower() in soft_skills:
            soft.append(token.text.lower())
    return {
        "technical_skills": tech,
        "soft_skills": list(set(soft))
    }
print(extract_skills("Python and SQL with good communication"))
#{'technical_skills': [], 'soft_skills': ['communication']}

text = "I know Python. I have worked with SQL and NLP."
result = extract_skills(text)
print(result)
#{'technical_skills': [], 'soft_skills': []}

tech_skills = ["Python", "SQL"]
patterns = [nlp(skill) for skill in tech_skills]
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
matcher.add("TECH", patterns)
text = "Experience in SQL and NoSQL"
doc = nlp(text)
skills = [doc[start:end].text.lower() for _, start, end in matcher(doc)]
print(skills)
#['sql']

resume = "Python, SQL, good communication"
job_desc = "Need NLP, SQL and teamwork"
resume_skills = extract_skills(resume)
job_skills = extract_skills(job_desc)
resume_json = json.dumps(resume_skills, indent=4)
job_json = json.dumps(job_skills, indent=4)
print("Resume Skills JSON:")
print(resume_json)
print("\nJob Skills JSON:")
print(job_json)
#Resume Skills JSON:
#{
#    "technical_skills": [
#        "sql",
#        "python"
#    ],
#    "soft_skills": [
#        "communication"
#    ]
#}

#Job Skills JSON:
#{
#    "technical_skills": [
#        "sql"
#    ],
#    "soft_skills": [
#        "teamwork"
#    ]
#}