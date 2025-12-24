tech_skills = ["Python", "Java", "SQL", "HTML", "Git"]
print(tech_skills)
#['Python', 'Java', 'SQL', 'HTML', 'Git']

soft_skills = ["Communication", "Teamwork", "Problem Solving", "Time Management", "Adaptability"]
print(soft_skills)
#['Communication', 'Teamwork', 'Problem Solving', 'Time Management', 'Adaptability']

resume_text = "Experienced in Python and SQL"
print(resume_text.lower())
#experienced in python and sql

resume_text = "Experienced in Python and SQL"
if "python" in resume_text.lower():
    print("Python exists")
else:
    print("Python does not exist")
#Python exists

tech_skills = ["Python", "SQL", "Java"]
soft_skills = ["Communication", "Teamwork"]
print("Technical Skills:")
for skill in tech_skills:
    print(skill)
print("Soft Skills:")
for skill in soft_skills:
    print(skill)
#Technical Skills:
#Python
#SQL
#Java
#Soft Skills:
#Communication
#Teamwork

resume_text = "I know Python, SQL and HTML"
tech_list = ["Python", "Java", "SQL", "HTML", "C"]
found_skills = []
for skill in tech_list:
    if skill.lower() in resume_text.lower():
        found_skills.append(skill)
print(found_skills)
#['Python', 'SQL', 'HTML']

job_desc = "Good communication and teamwork skills required"
soft_list = ["Communication", "Teamwork", "Leadership"]
found_soft = []
for skill in soft_list:
    if skill.lower() in job_desc.lower():
        found_soft.append(skill)
print(found_soft)
#['Communication', 'Teamwork']

import csv
with open("skills.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        if row[1] == "Technical":
            print(row[0])

resume_skills = ["Python", "SQL", "HTML"]
job_skills = ["Python", "Java", "SQL"]
common_skills = []
for skill in resume_skills:
    if skill in job_skills:
        common_skills.append(skill)
print(common_skills)
#['Python', 'SQL']

skills = ["Python", "SQL", "Python", "HTML"]
unique_skills = list(set(skills))
print(unique_skills)
#['Python', 'SQL', 'HTML']

import csv
skills_dict = {
    "Technical": [],
    "Soft": []
}
with open("skills.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        skill = row[0]
        category = row[1]

        if category in skills_dict:
            skills_dict[category].append(skill)
print(skills_dict)

resume_skills = ["Python", "SQL"]
job_skills = ["Python", "SQL", "Java"]
missing_skills = []
for skill in job_skills:
    if skill not in resume_skills:
        missing_skills.append(skill)
print(missing_skills)
#['Java']

resume_text = "Python SQL Python HTML Python"
skills = ["Python", "SQL", "HTML"]
for skill in skills:
    count = resume_text.count(skill)
    print(skill, ":", count)
#Python : 3
#SQL : 1
#HTML : 1

import csv
resume_text = "experienced in python and sql"
with open("skills.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        if row[0].lower() in resume_text.lower():
            print(row[0])

import csv
extracted_skills = ["Python", "SQL", "HTML"]
with open("extracted_skills.csv", "w", newline="") as file:
    writer = csv.writer(file)
    for skill in extracted_skills:
        writer.writerow([skill])
print("Skills saved to CSV")
