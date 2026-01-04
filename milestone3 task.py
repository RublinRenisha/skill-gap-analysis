resume_skills = ["Python", "SQL", "Machine Learning", "Python", "Data Analysis"]
jd_skills = ["Python", "Deep Learning", "SQL", "Deep Learning"]
resume_unique = list(set(resume_skills))
jd_unique = list(set(jd_skills))
print("Resume Skills (Unique):", resume_unique)
print("JD Skills (Unique):", jd_unique)
#Resume Skills (Unique): ['Data Analysis', 'Machine Learning', 'SQL', 'Python']
#JD Skills (Unique): ['Deep Learning', 'SQL', 'Python']

skills = [" Python ", "SQL", "  Machine Learning ", "Data Analysis  "]
cleaned_skills = [skill.strip().lower() for skill in skills]
print("Cleaned Skills:", cleaned_skills)
#Cleaned Skills: ['python', 'sql', 'machine learning', 'data analysis']

resume_skills = ["python", "sql", "machine learning"]
jd_skills = ["python", "deep learning", "statistics"]
skills_dict = {
    "resume_skills": resume_skills,
    "job_description_skills": jd_skills
}
print(skills_dict)
#{
 # 'resume_skills': ['python', 'sql', 'machine learning'],
  #'job_description_skills': ['python', 'deep learning', 'statistics']
#}

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully")

model = SentenceTransformer("all-MiniLM-L6-v2")
skill = "machine learning"
embedding = model.encode(skill)
print(embedding.shape)

model=SentenceTransformer("all-MiniLM-L6-v2")
resume_skills=["python","sql","machine learning","data analysis"]
embeddings=model.encode(resume_skills)
print(embeddings.shape)

model=SentenceTransformer("all-MiniLM-L6-v2")
jd_skills=["python","deep learning","statistics"]
embeddings=model.encode(jd_skills)
print(embeddings.shape)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model=SentenceTransformer("all-MiniLM-L6-v2")
skill1="machine learning"
skill2="deep learning"
emb1=model.encode([skill1])
emb2=model.encode([skill2])
similarity=cosine_similarity(emb1,emb2)
print(similarity[0][0])

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model=SentenceTransformer("all-MiniLM-L6-v2")
resume_skill="machine learning"
jd_skills=["deep learning","python","statistics"]
resume_emb=model.encode([resume_skill])
jd_emb=model.encode(jd_skills)
scores=cosine_similarity(resume_emb,jd_emb)[0]
for skill,score in zip(jd_skills,scores):
    print(skill,score)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model=SentenceTransformer("all-MiniLM-L6-v2")
resume_skills=["python","sql","machine learning"]
jd_skills=["python","deep learning","statistics"]
resume_emb=model.encode(resume_skills)
jd_emb=model.encode(jd_skills)
matrix=cosine_similarity(resume_emb,jd_emb)
print(matrix)

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model=SentenceTransformer("all-MiniLM-L6-v2")
resume_skills=["python","sql","machine learning"]
jd_skills=["python","deep learning","statistics"]
resume_emb=model.encode(resume_skills)
jd_emb=model.encode(jd_skills)
matrix=cosine_similarity(resume_emb,jd_emb)
df=pd.DataFrame(matrix,index=resume_skills,columns=jd_skills)
print(df)

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model=SentenceTransformer("all-MiniLM-L6-v2")
resume_skills=["python","sql","machine learning"]
jd_skills=["python","deep learning","statistics"]
resume_emb=model.encode(resume_skills)
jd_emb=model.encode(jd_skills)
matrix=cosine_similarity(resume_emb,jd_emb)
df=pd.DataFrame(matrix,index=resume_skills,columns=jd_skills)
for jd in df.columns:
    best_skill=df[jd].idxmax()
    best_score=df[jd].max()
    print(jd,best_skill,best_score)

high=0.75
mid=0.5
for jd in df.columns:
    score=df[jd].max()
    if score>=high:
        print(jd,"matched")
    elif score>=mid:
        print(jd,"partially matched")
    else:
        print(jd,"missing")

report={"matched":[],"partial":[],"missing":[]}
high=0.75
mid=0.5
for jd in df.columns:
    score=df[jd].max()
    best=df[jd].idxmax()
    if score>=high:
        report["matched"].append({"jd_skill":jd,"resume_skill":best,"score":float(score)})
    elif score>=mid:
        report["partial"].append({"jd_skill":jd,"resume_skill":best,"score":float(score)})
    else:
        report["missing"].append(jd)
print(report)

import json
with open("skill_gap_report.json","w") as f:
    json.dump(report,f,indent=4)
print("Report saved")

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df,annot=True,cmap="coolwarm")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
sns.heatmap(df,annot=True,cmap="coolwarm")
plt.xlabel("Job Description Skills")
plt.ylabel("Resume Skills")
plt.title("Skill Similarity Heatmap")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
sns.heatmap(df,annot=True,cmap="coolwarm")
for col in range(df.shape[1]):
    row=df.iloc[:,col].values.argmax()
    plt.scatter(col+0.5,row+0.5,s=200,facecolors='none',edgecolors='black')
plt.xlabel("Job Description Skills")
plt.ylabel("Resume Skills")
plt.title("Skill Similarity Heatmap with Highlights")
plt.show()

resume_skills=[]
jd_skills=["python","sql"]
if not resume_skills or not jd_skills:
    raise ValueError("Resume skills or Job Description skills cannot be empty")

def normalize_skill(skill):
    replacements={"ml":"machine learning","dl":"deep learning","ai":"artificial intelligence"}
    s=skill.strip().lower()
    return replacements.get(s,s)
skills=["ML","Python","AI","DL"]
normalized_skills=[normalize_skill(s) for s in skills]
print(normalized_skills)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
skills=["python","machine learning","sql"]
model1=SentenceTransformer("all-MiniLM-L6-v2")
model2=SentenceTransformer("all-MPNet-base-v2")
emb1=model1.encode(skills)
emb2=model2.encode(skills)
sim1=cosine_similarity(emb1,emb1)
sim2=cosine_similarity(emb2,emb2)
print("Similarity with model1:\n",sim1)
print("Similarity with model2:\n",sim2)

from sentence_transformers import SentenceTransformer
model=SentenceTransformer("all-MiniLM-L6-v2")
skills=["python","sql","python","ml"]
cache={}
embeddings=[]
for skill in skills:
    if skill not in cache:
        cache[skill]=model.encode(skill)
    embeddings.append(cache[skill])
print("Cached embeddings for skills:",list(cache.keys()))

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
def extract_skills(text):
    return [s.strip().lower() for s in text.split(",") if s.strip()]
def skill_gap_pipeline(resume_text,jd_text):
    resume_skills=extract_skills(resume_text)
    jd_skills=extract_skills(jd_text)
    model=SentenceTransformer("all-MiniLM-L6-v2")
    resume_emb=model.encode(resume_skills)
    jd_emb=model.encode(jd_skills)
    df=pd.DataFrame(cosine_similarity(resume_emb,jd_emb),index=resume_skills,columns=jd_skills)
    high=0.75
    mid=0.5
    report={"matched":[],"partial":[],"missing":[]}
    for jd in df.columns:
        score=df[jd].max()
        best=df[jd].idxmax()
        if score>=high:
            report["matched"].append({"jd_skill":jd,"resume_skill":best,"score":float(score)})
        elif score>=mid:
            report["partial"].append({"jd_skill":jd,"resume_skill":best,"score":float(score)})
        else:
            report["missing"].append(jd)
    return report
resume_text="Python, SQL, Machine Learning, Data Analysis"
jd_text="Python, Deep Learning, Statistics"
report=skill_gap_pipeline(resume_text,jd_text)
print(json.dumps(report,indent=4))

resume_skills=["python","sql","machine learning","data analysis"]
jd_skills=["python","deep learning","statistics"]
model=SentenceTransformer("all-MiniLM-L6-v2")
resume_emb=model.encode(resume_skills)
jd_emb=model.encode(jd_skills)
df=pd.DataFrame(cosine_similarity(resume_emb,jd_emb),index=resume_skills,columns=jd_skills)
top3_matches={jd:df[jd].sort_values(ascending=False).head(3).to_dict() for jd in df.columns}
print(top3_matches)

thresholds={"technical":0.75,"soft":0.5}
skills_type={"python":"technical","sql":"technical","communication":"soft"}
scores={"python":0.8,"sql":0.6,"communication":0.4}
for skill,score in scores.items():
    t=skills_type[skill]
    if score>=thresholds[t]:
        print(skill,"matched")
    elif score>=thresholds[t]*0.8:
        print(skill,"partially matched")
    else:
        print(skill,"missing")

resume_skills=["python","sql","machine learning"]
jd_skills=["python","deep learning","statistics"]
model=SentenceTransformer("all-MiniLM-L6-v2")
resume_emb=model.encode(resume_skills)
jd_emb=model.encode(jd_skills)
matrix=cosine_similarity(resume_emb,jd_emb)
alignment_score=float(np.max(matrix,axis=0).mean())
print("Overall Alignment Score:",alignment_score)

resume_emb=model.encode(resume_skills)
jd_emb=model.encode(jd_skills)
matrix=cosine_similarity(resume_emb,jd_emb)
df=pd.DataFrame(matrix,index=resume_skills,columns=jd_skills)
high=0.75
mid=0.5
report={"matched":[],"partial":[],"missing":[]}
for jd in df.columns:
    score=df[jd].max()
    best=df[jd].idxmax()
    if score>=high:
        report["matched"].append({"jd_skill":jd,"resume_skill":best,"score":float(score)})
    elif score>=mid:
        report["partial"].append({"jd_skill":jd,"resume_skill":best,"score":float(score)})
    else:
        report["missing"].append(jd)
with open("skill_gap_report.json","w") as f:
    json.dump(report,f,indent=4)
plt.figure(figsize=(8,6))
sns.heatmap(df,annot=True,cmap="coolwarm")
plt.xlabel("Job Description Skills")
plt.ylabel("Resume Skills")
plt.title("Skill Similarity Heatmap")
plt.savefig("skill_gap_heatmap.png")
plt.close()
print("Report and heatmap saved")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
class SkillGapAnalyzer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model=SentenceTransformer(model_name)
        self.cache={}
    def embed_skills(self,skills):
        embeddings=[]
        for skill in skills:
            if skill not in self.cache:
                self.cache[skill]=self.model.encode(skill)
            embeddings.append(self.cache[skill])
        return embeddings
    def compute_similarity(self,resume_skills,jd_skills):
        resume_emb=self.embed_skills(resume_skills)
        jd_emb=self.embed_skills(jd_skills)
        return pd.DataFrame(cosine_similarity(resume_emb,jd_emb),index=resume_skills,columns=jd_skills)
    def generate_report(self,df,high=0.75,mid=0.5):
        report={"matched":[],"partial":[],"missing":[]}
        for jd in df.columns:
            score=df[jd].max()
            best=df[jd].idxmax()
            if score>=high:
                report["matched"].append({"jd_skill":jd,"resume_skill":best,"score":float(score)})
            elif score>=mid:
                report["partial"].append({"jd_skill":jd,"resume_skill":best,"score":float(score)})
            else:
                report["missing"].append(jd)
        return report
resume_skills=["python","sql","machine learning"]
jd_skills=["python","deep learning","statistics"]
analyzer=SkillGapAnalyzer()
df=analyzer.compute_similarity(resume_skills,jd_skills)
report=analyzer.generate_report(df)
print(df)
print(json.dumps(report,indent=4))
