skills = ["Python","Java","C","GIT","Eclipse","Rapid Miner","Apache Mahout","Hive","MapReduce","MongoDB","Excel","SAS","R"]
Required_skills = ["Python","R","SQL","Machine Learning","Statistics","Data Cleaning","Data Visualization","Pandas","NumPy","Scikit-learn","TensorFlow","PyTorch","Matplotlib","Seaborn","Excel","Power BI","Tableau","Big Data Tools","Hadoop","Hive","Spark","MongoDB","Git","AWS","Azure","Communication Skills","Problem Solving"]
skills_have = [s.strip().lower() for s in skills]
skills_required= [r.strip().lower() for r in Required_skills]
skill_match=[]
skill_miss=[]
for i in skills_required:
    if i in skills_have:
        skill_match.append(i)
    else:
        skill_miss.append(i)
print("Matching skills: ",skill_match)
print("Missing Skills: ",skill_miss)